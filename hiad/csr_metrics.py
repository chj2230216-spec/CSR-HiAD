"""Streaming native-resolution metrics; fixed log1p bins, approximate ranks."""
import numpy as np


class PixelHistogram:
    def __init__(self, bins=65536, log_max=30.0):
        if bins<2 or log_max<=0:
            raise ValueError("invalid histogram")
        self.bins,self.log_max=bins,log_max
        self.pos=np.zeros(bins,np.int64)
        self.neg=np.zeros(bins,np.int64)
        self.pro=np.zeros(bins,np.float64)
        self.small=np.zeros(bins,np.float64)
        self.instances=self.small_instances=self.overflow=0

    def update(self,score,mask,labels,sizes):
        s=np.asarray(score)
        mask=np.asarray(mask,dtype=bool)
        if s.shape!=mask.shape or labels.shape!=mask.shape:
            raise ValueError("map/mask shape mismatch")
        if not np.isfinite(s).all() or (s<0).any():
            raise ValueError("invalid score")
        lx=np.log1p(s)
        self.overflow+=int((lx>=self.log_max).sum())
        ix=np.minimum((np.minimum(lx,self.log_max)*self.bins/self.log_max).astype(np.int32),self.bins-1)
        self.pos+=np.bincount(ix[mask],minlength=self.bins)
        self.neg+=np.bincount(ix[~mask],minlength=self.bins)
        self.instances+=len(sizes)-1
        ids=labels[mask]
        if len(ids):
            w=1.0/sizes[ids]
            self.pro+=np.bincount(ix[mask],weights=w,minlength=self.bins)
            small=sizes[ids]<=419
            self.small+=np.bincount(ix[mask][small],weights=w[small],minlength=self.bins)
            self.small_instances+=int(np.count_nonzero(sizes[1:]<=419))

    def metrics(self):
        p,n=self.pos.sum(),self.neg.sum()
        r=dict(positive_pixels=int(p),negative_pixels=int(n),instances=self.instances,
               small_instances=self.small_instances,histogram_bins=self.bins,
               log_score_max=self.log_max,overflow_pixels=self.overflow)
        if not p or not n:
            return dict(r,P_AUC=None,P_AP=None,AUPRO_030=None,small_PRO_FPR001=None,small_PRO_FPR01=None)
        tp=np.r_[0,np.cumsum(self.pos[::-1])]
        fp=np.r_[0,np.cumsum(self.neg[::-1])]
        recall=tp/p
        fpr=fp/n
        precision=np.divide(tp,tp+fp,out=np.ones_like(recall),where=(tp+fp)>0)
        r["P_AUC"]=float(np.sum(np.diff(fpr)*(recall[:-1]+recall[1:])*.5))
        r["P_AP"]=float(np.sum(np.diff(recall)*precision[1:]))
        r["AUPRO_030"]=None
        if self.instances:
            y=np.r_[0,np.cumsum(self.pro[::-1])]/self.instances
            dx=np.diff(fpr)
            frac=np.clip(np.divide(.3-fpr[:-1],dx,out=np.zeros_like(dx),where=dx>0),0,1)
            r["AUPRO_030"]=float(np.sum(dx*frac*(y[:-1]+(y[1:]-y[:-1])*frac*.5))/.3)
        for key,t in (("small_PRO_FPR001",.001),("small_PRO_FPR01",.01)):
            r[key]=None
            if self.small_instances:
                y=np.r_[0,np.cumsum(self.small[::-1])]/self.small_instances
                r[key]=float(y[np.searchsorted(fpr,t,side="right")-1])
        return r
