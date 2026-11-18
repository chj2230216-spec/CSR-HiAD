"""Shared-row, normal-only CSR scoring experiment (two aligned scales).

No GPU/backbone dependency here. All banks use identical sampled row IDs;
the diagnostic is not the original HiAD Coreset deployment protocol.
"""
import numpy as np
from hiad.csr_model import CSRTransform

MODES = ("fine", "fused", "paired_fine", "paired_fused", "concat",
         "residual", "conditional", "csr", "concat_std", "residual_std",
         "direct_csr", "raw_residual_score", "innovation_score",
         "direct_ridge_score", "coarse_bank", "innovation_bank", "separate_sum")


class IndexedReservoir:
    """Bounded uniform row sample, preserving global integer IDs."""
    def __init__(self, capacity, seed):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.rows = self.ids = self.keys = None
        self.count = 0

    def add(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("expected finite matrix")
        if len(x) == 0:
            return
        ids = np.arange(self.count, self.count+len(x), dtype=np.int64)
        self.count += len(x)
        keys = self.rng.random(len(x))
        if len(x) > self.capacity:
            i = np.argpartition(keys, -self.capacity)[-self.capacity:]
            x, ids, keys = x[i], ids[i], keys[i]
        if self.rows is not None:
            x = np.concatenate((self.rows, x))
            ids = np.concatenate((self.ids, ids))
            keys = np.concatenate((self.keys, keys))
        if len(x) > self.capacity:
            i = np.argpartition(keys, -self.capacity)[-self.capacity:]
            x, ids, keys = x[i], ids[i], keys[i]
        self.rows, self.ids, self.keys = x.copy(), ids.copy(), keys.copy()

    def value(self):
        if self.rows is None:
            raise ValueError("empty reservoir")
        i = np.argsort(self.ids)
        return self.rows[i], self.ids[i]


def unpack(rows):
    """[native fine D, native fused D, paired fine D/2, paired coarse D/2]."""
    x = np.asarray(rows, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % 6:
        raise ValueError("invalid packed features")
    f, fused, pair = np.split(x, 3, axis=1)
    return f, fused, pair.reshape(len(x), 2, -1)


class DiagnosticModel:
    def __init__(self, ridge=0.01, floor=0.01):
        self.ridge, self.floor = ridge, floor

    def fit(self, rows):
        _, _, x = unpack(rows)
        self.transforms = {m: CSRTransform(m, self.ridge, self.floor).fit(x)
                           for m in CSRTransform.MODES}
        # Holdout statistics are NOT used to fit any transforms.
        return self

    def representations(self, rows):
        f, fused, x = unpack(rows)
        d = x.shape[-1]
        z = {m: t.transform(x) for m,t in self.transforms.items()}
        z.update(fine=f, fused=fused, paired_fine=x[:,0],
                 paired_fused=(x[:,0]+x[:,1])*0.5)
        # Use exactly the same standardized blocks as joint CSR.
        z["coarse_bank"], z["innovation_bank"] = z["csr"][:,:d], z["csr"][:,d:]
        return {k: np.ascontiguousarray(v, dtype=np.float32) for k,v in z.items()}

    def direct_scores(self, z):
        d = z["csr"].shape[1] // 2
        return {"raw_residual_score": (z["residual_std"][:,d:]**2).mean(1),
                "innovation_score": (z["csr"][:,d:]**2).mean(1),
                "direct_ridge_score": (z["direct_csr"][:,d:]**2).mean(1)}

    def diagnostics(self, fit_rows, held_rows):
        _, _, x = unpack(fit_rows)
        _, _, h = unpack(held_rows)
        f, c = h[:,0], h[:,1]
        d = f.shape[1]
        baseline = ((f - x[:,0].mean(0))**2).sum()
        result = {}
        for m in ("csr", "direct_csr"):
            t = self.transforms[m]
            e = t._raw(h)[:,d:]
            bmat = t.weights[0] + (np.eye(d) if m == "csr" else 0)
            train_e = t._raw(x)[:,d:]
            result[m] = dict(holdout_r2=float(1-(e**2).sum()/baseline) if baseline>0 else None,
                             holdout_mse=float((e**2).mean()),
                             train_mse=float((train_e**2).mean()),
                             floor_fraction=float((train_e.std(0)<=self.floor).mean()),
                             B_frobenius=float(np.linalg.norm(bmat)),
                             B_minus_I_frobenius=float(np.linalg.norm(bmat-np.eye(d))))
        return result

    def state_dict(self):
        return {m:t.state_dict() for m,t in self.transforms.items()}


def mean_knn_numpy(query, bank, k):
    """Small-array exact reference for tests, not production-scale retrieval."""
    if not 1 <= k <= len(bank):
        raise ValueError("invalid k")
    dist = ((query[:,None,:]-bank[None,:,:])**2).sum(-1)
    return np.partition(dist,k-1,axis=1)[:,:k].mean(1)


def score_all(model, rows, search):
    """search(name, query) returns mean squared L2 of k nearest bank rows."""
    z = model.representations(rows)
    scores = {name: search(name, q) for name,q in z.items()}
    scores.update(model.direct_scores(z))
    scores["separate_sum"] = scores["coarse_bank"] + scores["innovation_bank"]
    return {m: np.asarray(scores[m], dtype=np.float32) for m in MODES}
