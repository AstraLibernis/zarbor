#!/usr/bin/env python
"""Pairs two TSV runs and reports mean delta vs sd across seeds."""
import sys, re, collections
def load(p):
    d=collections.defaultdict(dict)
    for line in open(p):
        f=line.rstrip("\n").split("\t")
        if len(f)<5 or "oof" not in f[4]: continue
        m=re.search(r"oof\s+([0-9.]+)", f[4]); ms=re.search(r"(\d+) ms", f[4])
        d[f[1]][int(f[2])]=(float(m.group(1)), int(ms.group(1)) if ms else 0)
    return d
a,b=load(sys.argv[1]),load(sys.argv[2])
lower_is_better={"california":1,"ames":1,"housing":1,"adult":0,"bank":0}
print(f"{'dataset':11s} {'base':>13s} {'new':>13s} {'delta':>11s} {'sd':>9s} "
      f"{'|d|/sd':>7s}  {'ms':>10s}  verdict")
for k in a:
    if k not in b: continue
    seeds=sorted(set(a[k])&set(b[k]))
    da=[b[k][s][0]-a[k][s][0] for s in seeds]
    n=len(da); mu=sum(da)/n
    sd=(sum((x-mu)**2 for x in da)/(n-1))**.5 if n>1 else float('nan')
    base=sum(a[k][s][0] for s in seeds)/n; new=sum(b[k][s][0] for s in seeds)/n
    tms=sum(a[k][s][1] for s in seeds)/n; nms=sum(b[k][s][1] for s in seeds)/n
    good = (mu<0) if lower_is_better[k] else (mu>0)
    sig = sd>0 and abs(mu) > 2*sd
    v = ("HELPS" if good else "HURTS") if sig else ("identical" if mu==0 else "no effect")
    print(f"{k:11s} {base:13.6f} {new:13.6f} {mu:+11.6f} {sd:9.6f} "
          f"{(abs(mu)/sd if sd>0 else 0):7.2f}  {nms/tms:9.2f}x  {v}  (n={n})")
