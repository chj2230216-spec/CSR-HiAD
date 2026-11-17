"""CSR normal-only feature modeling; no dependency on the HiAD backbone."""
import numpy as np


class Reservoir:
    """Uniform bounded row sample via random priorities; deterministic per seed."""
    def __init__(self, capacity, seed=42):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.rows = None
        self.keys = np.empty(0, dtype=np.float64)

    def add(self, rows):
        rows = np.asarray(rows, dtype=np.float32)
        if not len(rows):
            return
        if not np.isfinite(rows).all():
            raise ValueError("non-finite features")
        keys = self.rng.random(len(rows))
        if len(keys) > self.capacity:
            idx = np.argpartition(keys, -self.capacity)[-self.capacity:]
            rows, keys = rows[idx], keys[idx]
        if self.rows is not None:
            rows = np.concatenate([self.rows, rows])
            keys = np.concatenate([self.keys, keys])
        if len(keys) > self.capacity:
            idx = np.argpartition(keys, -self.capacity)[-self.capacity:]
            rows, keys = rows[idx], keys[idx]
        self.rows, self.keys = rows.copy(), keys.copy()

    def value(self):
        if self.rows is None:
            raise ValueError("empty reservoir")
        return self.rows


def ridge_fit(c, r, ridge=0.01):
    """Solve mean squared loss + ridge * ||W||_F^2, row-vector convention."""
    if ridge <= 0 or len(c) < 2:
        raise ValueError("positive ridge and >=2 rows required")
    c, r = np.asarray(c, dtype=np.float64), np.asarray(r, dtype=np.float64)
    mc, mr = c.mean(0), r.mean(0)
    x, y = c - mc, r - mr
    gram = x.T @ x / len(x) + ridge * np.eye(x.shape[1])
    w = np.linalg.solve(gram, x.T @ y / len(x))
    return w.astype(np.float32), (mr - mc @ w).astype(np.float32)


class CSRTransform:
    """Input [N, S, d], ordered finest to coarsest; output [N, S*d].
    residual: [coarsest, fine-coarse differences].
    conditional: predict each adjacent residual from the next-coarser scale.
    csr: conditional innovations plus normal-data diagonal standardization.
    No target fine feature enters its predictor.
    """
    MODES = ("concat", "residual", "conditional", "csr")

    def __init__(self, mode="csr", ridge=0.01, std_floor=0.01):
        if mode not in self.MODES or ridge <= 0 or std_floor <= 0:
            raise ValueError("invalid CSR configuration")
        self.mode, self.ridge, self.std_floor = mode, ridge, std_floor
        self.weights, self.biases = [], []
        self.mean = self.std = None
        self.shape = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1] < 2 or len(x) < 2:
            raise ValueError("expected >=2 feature rows and >=2 aligned scales")
        self.shape = x.shape[1:]
        self.weights, self.biases = [], []
        if self.mode in ("conditional", "csr"):
            for k in range(x.shape[1] - 1):
                c = x[:, k+1]
                w, b = ridge_fit(c, x[:, k] - c, self.ridge)
                self.weights.append(w)
                self.biases.append(b)
        if self.mode == "csr":
            z = self._raw(x)
            self.mean = z.mean(0)
            self.std = np.maximum(z.std(0), self.std_floor)
        return self

    def _raw(self, x):
        if x.ndim != 3 or x.shape[1:] != self.shape:
            raise ValueError("scale count or feature dimension mismatch")
        if self.mode == "concat":
            return x.reshape(len(x), -1)
        parts = [x[:, -1]]
        for k in range(x.shape[1] - 1):
            c = x[:, k+1]
            r = x[:, k] - c
            if self.mode in ("conditional", "csr"):
                r = r - (c @ self.weights[k] + self.biases[k])
            parts.append(r)
        return np.concatenate(parts, axis=1)

    def transform(self, x):
        z = self._raw(np.asarray(x, dtype=np.float32))
        if self.mode == "csr":
            z = (z - self.mean) / self.std
        return np.ascontiguousarray(z, dtype=np.float32)

    def state_dict(self):
        return dict(mode=self.mode, ridge=self.ridge, std_floor=self.std_floor,
                    shape=self.shape, weights=self.weights, biases=self.biases,
                    mean=self.mean, std=self.std)

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state["mode"], state["ridge"], state["std_floor"])
        for name in ("shape", "weights", "biases", "mean", "std"):
            setattr(obj, name, state[name])
        return obj


def tail_score(scores, sorted_normal):
    """Empirical survival rank; NOT a posterior probability or pixelwise p-value."""
    ref = np.asarray(sorted_normal)
    if ref.ndim != 1 or len(ref) == 0:
        raise ValueError("empty calibration reference")
    n = len(ref)
    count = n - np.searchsorted(ref, scores, side="left")
    return -np.log((1.0 + count) / (n + 1.0))
