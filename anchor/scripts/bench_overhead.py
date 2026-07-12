#!/usr/bin/env python3
"""Measure the ρ-anchor overhead on top of vanilla: forward, descriptors, kNN, pair-corr."""
import os, sys, time
from pathlib import Path
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rho_anchor_predict import Rho, pair_corr_gated

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
rho = Rho(); rho.r_lo, rho.r_hi = 3.0, 4.0
import torch

def tmean(fn, n=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/n*1000  # ms

for tag, path, idx in [("keep ~%d at", PRE/"splits"/"keep_test.xyz", 0),
                       ("MPtrj ~%d at", os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), 0)]:
    at = read(path, index=str(idx)); nat=len(at)
    def f_van():
        at.calc=calc; return at.get_potential_energy(), at.get_forces()
    def f_desc(): return np.asarray(calc.get_descriptors(at))
    desc = f_desc(); r = rho.of(desc)
    def f_knn(): return rho.of(np.asarray(desc))           # PCA proj + kNN
    def f_pair(): return pair_corr_gated(at, r, 800,0.5,0.3,1.5,2)
    t_van=tmean(f_van); t_desc=tmean(f_desc); t_knn=tmean(f_knn); t_pair=tmean(f_pair,100)
    print(f"\n=== {tag % nat} ===")
    print(f"  vanilla E+F (1 forward)   : {t_van:8.2f} ms   [baseline]")
    print(f"  + descriptors (2nd fwd)   : {t_desc:8.2f} ms   ({t_desc/t_van*100:.0f}% baseline)")
    print(f"  + rho PCA+kNN             : {t_knn:8.2f} ms   ({t_knn/t_van*100:.1f}% baseline)")
    print(f"  + pair-corr (BM+nl)       : {t_pair:8.2f} ms   ({t_pair/t_van*100:.1f}% baseline)")
    naive=t_van+t_desc+t_knn+t_pair; fused=t_van+t_knn+t_pair
    print(f"  TOTAL naive (as in script): {naive:8.2f} ms   = x{naive/t_van:.2f} vanilla")
    print(f"  TOTAL fused (1 forward)   : {fused:8.2f} ms   = x{fused/t_van:.2f} vanilla")
