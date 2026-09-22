#!/usr/bin/env python
"""Runs the pre-registered grid in docs/PROTOCOL.md. Emits TSV on stdout."""
import subprocess, sys, time, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED = ["--n_rounds=500","--learning_rate=0.05","--max_depth=6",
         "--min_child_samples=20","--lambda=1.0","--max_bin=256"]
GRID = [("california","y","squared_error",[]),
        ("adult","y","logistic",[]),
        ("bank","y","logistic",[]),
        ("ames","y","squared_error",[]),
        ("housing","AffordabilityPercentageTrue","squared_error",["--group-col=Zip","--drop=City","--drop=Metro","--drop=Zip3"])]

def main():
    binary, tag = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]
    only = os.environ.get("ONLY")
    for name, label, obj, ds_extra in GRID:
        if only and name not in only.split(","): continue
        for seed in (0,1,2):
            cmd = [binary,"cv",f"{ROOT}/bench/data/{name}.csv",f"--label={label}",
                   "--folds=5",f"--fold-seed={seed}",f"--objective={obj}",
                   *FIXED,*ds_extra,*extra,"--quiet=1"]
            t0=time.time()
            r=subprocess.run(cmd,capture_output=True,text=True)
            dt=time.time()-t0
            line=(r.stdout+r.stderr).strip().replace("\n"," | ")
            print(f"{tag}\t{name}\t{seed}\t{dt:8.2f}\t{line}", flush=True)
main()
