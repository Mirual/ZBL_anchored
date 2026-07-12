import os, sys, time
from pathlib import Path
import numpy as np, torch
from ase.io import read
from mace.calculators import MACECalculator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rho_anchor_predict import Rho, pair_corr_gated
VAN=os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
PRE=Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
calc=MACECalculator(model_paths=[VAN],device="cuda",default_dtype="float32",head="mp_pbe_refit_add")
rho=Rho(); rho.r_lo,rho.r_hi=3.0,4.0
def tmean(fn,n=30,w=8):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
c=[0]
for tag,path in [("keep",PRE/"splits"/"keep_test.xyz"),
                 ("MPtrj",os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"))]:
    at=read(path,index="0"); nat=len(at)
    def f_van():                                  # cache-bust: fresh copy + micro-shift
        c[0]+=1; a=at.copy(); a.positions[0,0]+=1e-6*(c[0]%7); a.calc=calc
        return a.get_potential_energy(), a.get_forces()
    desc=np.asarray(calc.get_descriptors(at)); r=rho.of(desc)
    def f_desc(): return np.asarray(calc.get_descriptors(at))
    def f_knn(): return rho.of(desc)
    def f_pair(): return pair_corr_gated(at,r,800,0.5,0.3,1.5,2)
    tv=tmean(f_van); td=tmean(f_desc); tk=tmean(f_knn); tp=tmean(f_pair,200)
    naive=tv+td+tk+tp; fused=tv+tk+tp
    print(f"\n=== {tag} ({nat} at) ===")
    print(f"  vanilla E+F (1 forward)        : {tv:7.2f} ms   [baseline]")
    print(f"  ρ: descriptors (2nd forward)   : {td:7.2f} ms   (x{td/tv:.2f})")
    print(f"  ρ: PCA proj + kNN bank         : {tk:7.2f} ms   ({tk/tv*100:.0f}%)")
    print(f"  pair-corr (neighbor+Born-Mayer): {tp:7.2f} ms   ({tp/tv*100:.0f}%)")
    print(f"  -> naive (as in script, 2 fwd) : {naive:7.2f} ms = x{naive/tv:.2f}")
    print(f"  -> fused (1 fwd, reuse feats)  : {fused:7.2f} ms = x{fused/tv:.2f}")
