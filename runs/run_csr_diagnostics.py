"""Shared-feature Round 1 diagnostic, fixed paired rows, A2O, native 2K.

Run from repository root. No synthetic anomalies, SCA, thumbnail, or tail
calibration. This is a fixed-row diagnostic, not an upstream Coreset run.
"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml
import faiss
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from hiad.csr_diagnostics import DiagnosticModel, IndexedReservoir, MODES, score_all
from hiad.csr_metrics import PixelHistogram
from hiad.detectors.hr_csr_patchcore import HRCSRPatchCore
from hiad.utils.split_and_gather import HRSample, MultiResolutionHRImageSpliter


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False)+"\n")


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def read_meta(root,category,split):
    p=root/category/(split+".jsonl")
    data,seen=[],set()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r=json.loads(line)
        if r["clsname"]!=category or r["label"] not in (0,1) or (split=="train" and r["label"]!=0):
            raise ValueError("invalid category/label")
        img=(root/r["filename"]).resolve()
        if img in seen or root not in img.parents:
            raise ValueError("duplicate or outside-root image: "+str(img))
        seen.add(img)
        with Image.open(img) as im:
            if im.size!=(2048,2048):
                raise ValueError("expected native 2048x2048: "+str(img))
        r.update(path=str(img),sha256=sha(img))
        if r["label"] or r.get("mask"):
            mask=(root/r["mask"]).resolve()
            if root not in mask.parents:
                raise ValueError("mask outside root")
            with Image.open(mask) as m:
                if m.size!=(2048,2048) or bool(m.convert("L").getbbox())!=bool(r["label"]):
                    raise ValueError("mask/label mismatch: "+str(mask))
            r.update(mask_path=str(mask),mask_sha256=sha(mask))
        data.append(r)
    if not data:
        raise ValueError("empty manifest")
    return data,sha(p)


class SharedExtractor:
    def __init__(self,cfg,args):
        self.device=torch.device("cuda",args.gpu)
        self.det=HRCSRPatchCore(**cfg,patch_size=512,logger=logging.getLogger("round1"),
                    device=self.device,seed=args.seed,fusion_weights=[.5,.5],csr_mode="csr")
        self.det.forward_modules.eval()
        self.indices=MultiResolutionHRImageSpliter(2048,512,[0,1],512)
        if len(self.indices)!=16:
            raise ValueError("expected 16 patches")
        self.batch,self.dim=args.batch_size,args.dim
        self.check=args.audit_alignment

    @torch.no_grad()
    def extract(self,path):
        sample=HRSample(image=path,image_size=2048)
        sample.open()
        patches=[]
        try:
            for idx in self.indices:
                p=sample[idx.main_index]
                for low in idx.low_resolution_indexes:
                    p.add_low_resolution_images(low,sample.image)
                patches.append(p)
        finally:
            sample.close()
        loader=DataLoader(self.det.create_dataset(patches,False,"diagnostic"),
                          batch_size=self.batch,shuffle=False,num_workers=0,drop_last=False)
        chunks=[]
        grid=None
        for data in loader:
            scales=self.det.get_multi_resolution_aligned_embeddings(data)
            if len(scales)!=2:
                raise ValueError("expected two aligned scales")
            fine,ps=self.det._embed_scale(scales[0],detach=False,provide_patch_shapes=True)
            coarse,cps=self.det._embed_scale(scales[1],detach=False,provide_patch_shapes=True)
            fused,fps=self.det._embed_scale([(a+b)*.5 for a,b in zip(*scales)],detach=False,provide_patch_shapes=True)
            if ps!=cps or ps!=fps:
                raise ValueError("spatial shapes differ")
            if self.check:
                ref,rps=self.det._embed_scale(self.det.get_multi_resolution_fusion_embeddings(data),detach=False,provide_patch_shapes=True)
                torch.testing.assert_close(fused,ref,rtol=1e-4,atol=1e-5)
                if rps!=ps:
                    raise ValueError("upstream fusion shape mismatch")
                self.check=False
            grid=tuple(ps[0])
            if fine.shape!=coarse.shape or fine.shape!=fused.shape or fine.shape[1]!=self.dim:
                raise ValueError("embedding shape mismatch")
            if fine.shape[0]!=len(data["image"])*grid[0]*grid[1]:
                raise ValueError("row/grid mismatch")
            f=F.adaptive_avg_pool1d(fine[:,None,:],self.dim//2).squeeze(1)
            c=F.adaptive_avg_pool1d(coarse[:,None,:],self.dim//2).squeeze(1)
            packed=torch.cat((fine,fused,f,c),dim=1).cpu().numpy()
            if not np.isfinite(packed).all():
                raise ValueError("nonfinite features")
            chunks.append(packed)
        rows=np.concatenate(chunks)
        if len(rows)!=16*grid[0]*grid[1]:
            raise ValueError("missing rows")
        return rows,grid

    def map(self,score,grid,smooth):
        x=torch.from_numpy(score.reshape(16,1,*grid))
        patches=F.interpolate(x,size=(512,512),mode="bilinear",align_corners=False)[:,0].numpy()
        image=np.empty((2048,2048),np.float32)
        for p,idx in zip(patches,self.indices):
            m=idx.main_index
            if smooth:
                p=ndimage.gaussian_filter(p,sigma=4)
            image[m.y:m.y+m.height,m.x:m.x+m.width]=p
        return ndimage.gaussian_filter(image,sigma=4) if smooth else image


def image_score(m):
    return float(F.avg_pool2d(torch.from_numpy(m)[None,None],16,stride=1).max())


def run(args,out):
    root=Path(args.data_root).resolve()
    train,trsha=read_meta(root,args.category,"train")
    test,tesha=read_meta(root,args.category,"test")
    if set(r["path"] for r in train)&set(r["path"] for r in test) or set(r["sha256"] for r in train)&set(r["sha256"] for r in test):
        raise ValueError("train/test overlap")
    if len(train)<3 or {r["label"] for r in test}!={0,1}:
        raise ValueError("need >=3 normals and both test labels")
    order=np.random.default_rng(args.seed).permutation(len(train))
    n=max(1,min(len(train)-2,round(len(train)*args.holdout_ratio)))
    fit=[train[i] for i in sorted(order[n:])]
    hold=[train[i] for i in sorted(order[:n])]
    cfg=yaml.safe_load(Path(args.config).read_text())
    cfg["target_embed_dimension"]=args.dim
    cfg["faiss_on_gpu"]=False
    dump(out/"manifest.json",dict(args=vars(args),config=cfg,config_sha256=sha(Path(args.config)),
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip()),
        torch=torch.__version__,cuda=torch.version.cuda,faiss=getattr(faiss,"__version__","unknown"),
        gpu_name=torch.cuda.get_device_name(args.gpu),train_manifest_sha256=trsha,
        test_manifest_sha256=tesha,fit=fit,holdout=hold,test=test,modes=MODES,
        bank_policy="shared uniform rows; not Coreset",metric_protocol="fixed log1p histogram, native 2048"))
    extractor=SharedExtractor(cfg,args)
    hh=hashlib.sha256()
    for name,v in sorted(extractor.det.backbone.state_dict().items()):
        hh.update(name.encode())
        hh.update(v.detach().cpu().contiguous().numpy().tobytes())
    dump(out/"backbone.json",{"sha256":hh.hexdigest(),"patches":[str(idx) for idx in extractor.indices]})
    fr,br,hr=(IndexedReservoir(args.fit_rows,args.seed),IndexedReservoir(args.bank_rows,args.seed+1),
              IndexedReservoir(args.fit_rows,args.seed+2))
    ranges=[]
    t=time.perf_counter()
    for split,records in (("fit",fit),("holdout",hold)):
        offset=0
        for i,r in enumerate(records):
            x,grid=extractor.extract(r["path"])
            ranges.append(dict(split=split,path=r["path"],first=offset,stop=offset+len(x),grid=grid))
            offset+=len(x)
            if split=="fit":
                fr.add(x)
                br.add(x)
            else:
                hr.add(x)
            print(split,i+1,len(records),flush=True)
    xf,fi=fr.value()
    xb,bi=br.value()
    xh,hi=hr.value()
    if len(xb)<args.k:
        raise ValueError("insufficient bank rows")
    model=DiagnosticModel(args.ridge,args.floor).fit(xf)
    banks=model.representations(xb)
    dump(out/"row_ranges.json",ranges)
    np.savez_compressed(out/"shared_rows.npz",fit_ids=fi,bank_ids=bi,holdout_ids=hi)
    torch.save(dict(transforms=model.state_dict(),banks=banks),out/"diagnostic_state.pt")
    dump(out/"normal_diagnostics.json",model.diagnostics(xf,xh))
    del xf,xb,xh,fr,br,hr
    faiss.omp_set_num_threads(args.faiss_threads)
    indexes={}
    resource=faiss.StandardGpuResources() if args.faiss_gpu else None
    for name,bank in banks.items():
        idx=faiss.IndexFlatL2(bank.shape[1])
        idx.add(bank)
        if resource is not None:
            idx=faiss.index_cpu_to_gpu(resource,args.gpu,idx)
        indexes[name]=idx
    def search(name,q):
        ds,ids=indexes[name].search(q,args.k)
        if (ids<0).any() or not np.isfinite(ds).all():
            raise ValueError("invalid neighbors")
        return np.maximum(ds,0).mean(1)
    torch.cuda.synchronize(args.gpu)
    fit_sec=time.perf_counter()-t
    hist={m:PixelHistogram(args.bins) for m in MODES}
    records=[]
    pred=out/"predictions"
    pred.mkdir()
    for i,r in enumerate(test):
        x,grid=extractor.extract(r["path"])
        scores=score_all(model,x,search)
        if r["label"]:
            with Image.open(r["mask_path"]) as im:
                mask=np.asarray(im.convert("L"))!=0
        else:
            mask=np.zeros((2048,2048),bool)
        labels,_=ndimage.label(mask,structure=np.ones((3,3),int))
        sizes=np.bincount(labels.ravel())
        dest=pred/f"{i:04d}"
        dest.mkdir()
        np.savez_compressed(dest/"grid_scores.npz",**{m:scores[m].reshape(16,*grid) for m in MODES})
        if args.save_full:
            np.save(dest/"mask.npy",mask.astype(np.uint8))
        row=dict(image_id=i,path=r["path"],label=r["label"],label_name=r["label_name"])
        for m,s in scores.items():
            amap=extractor.map(s,grid,not args.no_smoothing)
            hist[m].update(amap,mask,labels,sizes)
            row[m]=image_score(amap)
            if args.save_full:
                np.save(dest/(m+".npy"),amap)
        records.append(row)
        print("test",i+1,len(test),flush=True)
    def write_csv(path,rs):
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    write_csv(out/"image_scores.csv",records)
    y=[r["label"] for r in records]
    metrics=[]
    for m,h in hist.items():
        a=[r[m] for r in records]
        metrics.append(dict(mode=m,category=args.category,seed=args.seed,
                       I_AUC=float(roc_auc_score(y,a)),I_AP=float(average_precision_score(y,a)),**h.metrics()))
        np.savez_compressed(out/(m+"_hist.npz"),pos=h.pos,neg=h.neg,pro=h.pro,small=h.small)
    dump(out/"metrics.json",metrics)
    write_csv(out/"metrics.csv",metrics)
    dump(out/"runtime.json",dict(fit_extract_and_index_sec=fit_sec,bank_rows=len(bi),fit_rows=len(fi),held_rows=len(hi),
         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.gpu),
         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(args.gpu),
         caveat="shared diagnostic timing; allocator excludes external FAISS allocations; not isolated deployment benchmark"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root",default="data/MVTec-2K")
    p.add_argument("--category",choices=("bottle","capsule"),required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--config",default="configs/patchcore.yaml")
    for name,value in (("gpu",0),("batch_size",2),("seed",42),("dim",384),("fit_rows",20000),("bank_rows",5000),("k",5),("bins",65536),("faiss_threads",4)):
        p.add_argument("--"+name,type=int,default=value)
    for name,value in (("ridge",.01),("floor",.01),("holdout_ratio",.2)):
        p.add_argument("--"+name,type=float,default=value)
    for name in ("faiss_gpu","audit_alignment","save_full","no_smoothing"):
        p.add_argument("--"+name,action="store_true")
    a=p.parse_args()
    if a.dim<2 or a.dim%2 or not 0<a.holdout_ratio<1 or min(a.batch_size,a.fit_rows,a.bank_rows,a.k)<1:
        p.error("invalid settings")
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    try:
        run(a,out)
        dump(out/"status.json",{"status":"complete"})
    except Exception as e:
        dump(out/"status.json",{"status":"failed","error":repr(e)})
        raise


if __name__=="__main__":
    main()
