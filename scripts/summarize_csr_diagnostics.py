"""Summarize successful shared diagnostics without treating failures as zeros."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

FIELDS=("I_AUC","I_AP","P_AUC","P_AP","AUPRO_030","small_PRO_FPR001","small_PRO_FPR01")
PAIRS=(("fused","fine"),("concat","fused"),("paired_fused","paired_fine"),
       ("concat","paired_fused"),("residual","concat"),("conditional","residual"),
       ("csr","conditional"),("csr","concat_std"),("csr","residual_std"),
       ("direct_csr","csr"),("innovation_score","raw_residual_score"),
       ("direct_ridge_score","innovation_score"),("csr","separate_sum"),
       ("innovation_bank","innovation_score"),("csr","fused"))
def write(path,rows,fields):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    root,out=Path(a.root),Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    rows,status,delta=[],[],[]
    group=defaultdict(list)
    for category in ("bottle","capsule"):
        for d in sorted((root/category).glob("seed*")):
            s=json.loads((d/"status.json").read_text()) if (d/"status.json").exists() else {"status":"incomplete"}
            status.append(dict(category=category,run=d.name,status=s["status"],error=s.get("error","")))
            if s["status"]!="complete":
                continue
            rs=json.loads((d/"metrics.json").read_text())
            rows.extend(rs)
            by={r["mode"]:r for r in rs}
            for m,b in PAIRS:
                for key in FIELDS:
                    x,y=by[m][key],by[b][key]
                    val=None if x is None or y is None else 100*(x-y)
                    rec=dict(category=category,seed=by[m]["seed"],mode=m,baseline=b,metric=key,delta_pp=val)
                    delta.append(rec)
                    if val is not None:
                        group[m,b,key,category].append(val)
    write(out/"status.csv",status,["category","run","status","error"])
    if not rows:
        raise SystemExit("No complete runs; status.csv saved")
    write(out/"metrics.csv",rows,list(rows[0]))
    write(out/"paired_deltas.csv",delta,["category","seed","mode","baseline","metric","delta_pp"])
    agg=[]
    for m,b,key in PAIRS:
        buckets=[group[m,b,key,c] for c in ("bottle","capsule")]
        vals=[mean(x) if x else None for x in buckets]
        agg.append(dict(mode=m,baseline=b,metric=key,bottle_mean_pp=vals[0],capsule_mean_pp=vals[1],
                        bottle_n=len(buckets[0]),capsule_n=len(buckets[1]),
                        two_category_macro_pp=mean(vals) if all(v is not None for v in vals) else None))
    write(out/"macro_deltas.csv",agg,list(agg[0]))
if __name__=="__main__":
    main()
