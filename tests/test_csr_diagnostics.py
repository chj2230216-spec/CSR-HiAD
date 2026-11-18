import numpy as np
from hiad.csr_model import CSRTransform
from hiad.csr_diagnostics import IndexedReservoir, DiagnosticModel, mean_knn_numpy, score_all, MODES


def rows(n=80):
    rng = np.random.default_rng(42)
    return rng.normal(size=(n,12)).astype(np.float32)


def test_shared_reservoir():
    x = rows()
    a, b = IndexedReservoir(12,5), IndexedReservoir(12,5)
    a.add(x)
    for chunk in np.array_split(x,7):
        b.add(chunk)
    va, ia = a.value()
    vb, ib = b.value()
    np.testing.assert_array_equal(ia,ib)
    np.testing.assert_array_equal(va,vb)
    np.testing.assert_array_equal(va,x[ia])


def test_standardization_and_direct_ridge():
    x = rows()[:,8:].reshape(-1,2,2)
    for m in ("concat_std","residual_std","direct_csr"):
        t = CSRTransform(m).fit(x)
        z = t.transform(x)
        np.testing.assert_allclose(z.mean(0),0,atol=1e-6)
        np.testing.assert_allclose(z.std(0),1,atol=1e-6)
        np.testing.assert_array_equal(CSRTransform.from_state_dict(t.state_dict()).transform(x),z)
    c = x[:,1].astype("float64")
    f = x[:,0].astype("float64")
    c -= c.mean(0)
    f -= f.mean(0)
    lam = 0.01
    identity = CSRTransform("csr",lam).fit(x).weights[0]+np.eye(2)
    direct = CSRTransform("direct_csr",lam).fit(x).weights[0]
    np.testing.assert_allclose(identity-direct,lam*np.linalg.inv(c.T@c/len(c)+lam*np.eye(2)),atol=1e-6)


def test_score_geometry():
    x = rows()
    m = DiagnosticModel().fit(x[:40])
    bank = m.representations(x[40:60])
    scores = score_all(m,x[60:],lambda n,q: mean_knn_numpy(q,bank[n],5))
    assert set(scores)==set(MODES)
    assert all(np.isfinite(v).all() and (v>=0).all() for v in scores.values())
    assert np.all(scores["csr"]+1e-5>=scores["separate_sum"])
    assert "holdout_r2" in m.diagnostics(x[:40],x[60:])["csr"]
