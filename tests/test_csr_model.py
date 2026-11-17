import numpy as np
from hiad.csr_model import CSRTransform, Reservoir, ridge_fit, tail_score

def test_ridge():
    rng = np.random.default_rng(1)
    c = rng.normal(size=(1000, 4))
    r = 2*c + 1
    w,b = ridge_fit(c,r,1e-6)
    assert np.max(np.abs(c@w+b-r)) < 1e-4

def test_residual_reconstruct():
    x = np.random.default_rng(0).normal(size=(10,3,4)).astype(np.float32)
    t = CSRTransform("residual").fit(x)
    c,r0,r1 = np.split(t.transform(x),3,axis=1)
    np.testing.assert_allclose(c+r1+r0,x[:,0],atol=1e-6)

def test_modes_and_checkpoint():
    x = np.random.default_rng(2).normal(size=(500,3,6)).astype(np.float32)
    for mode in CSRTransform.MODES:
        t = CSRTransform(mode).fit(x)
        z = t.transform(x)
        assert z.shape == (500,18)
        np.testing.assert_array_equal(CSRTransform.from_state_dict(t.state_dict()).transform(x), z)
    assert np.max(np.abs(z.mean(0))) < 1e-5

def test_reservoir_and_tail():
    r = Reservoir(8,1)
    for i in range(10):
        r.add(np.arange(10).reshape(-1,1)+10*i)
    assert r.value().shape == (8,1)
    t = tail_score(np.array([0,1,2,3]),np.array([1,2,2]))
    assert np.all(np.diff(t)>=0)
    assert t[0]==0 and np.isclose(t[-1],np.log(4))
