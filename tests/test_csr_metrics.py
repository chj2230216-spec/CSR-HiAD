import numpy as np
from hiad.csr_metrics import PixelHistogram

def test_perfect():
    m=np.array([[0,0],[1,1]],bool)
    h=PixelHistogram(128)
    h.update(m.astype(float)*10,m,m.astype(int),np.array([2,2]))
    r=h.metrics()
    assert r["P_AP"]==1 and r["P_AUC"]==1
    assert abs(r["AUPRO_030"]-1)<1e-10
    assert r["small_PRO_FPR01"]==1

def test_ties():
    m=np.array([[0,0],[1,1]],bool)
    h=PixelHistogram(128)
    h.update(np.zeros(m.shape),m,m.astype(int),np.array([2,2]))
    r=h.metrics()
    assert r["P_AP"]==0.5 and r["P_AUC"]==0.5
    assert r["small_PRO_FPR01"]==0
    assert abs(r["AUPRO_030"]-0.15)<1e-9
