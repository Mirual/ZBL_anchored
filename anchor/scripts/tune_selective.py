#!/usr/bin/env python3
"""SelectiveNet-style risk-coverage calibration of the RND gate (ported from selective prediction,
Geifman 1901.09192): instead of a manual threshold — optimization of θ=(r_lo,r_hi,A) on the objective

    min_θ  R_keep(θ)        # F-MAE on keep_train (where the correction helps)
    s.t.   ΔF_base(θ) ≤ ε   # increase of F-MAE on (u200_train ∪ MPtrj-calib) — the base is NOT degraded

Calibration on train/calib splits (disjoint from held-out tests). We cache (F_van, F_ref, novelty,
pairs) once → the grid over θ is instant. Best θ → results/selective_theta.json.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_predict import smoothstep, dsmoothstep
from rnd_anchor_predict import RNDGate

VAN = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight")) / "splits"
MIXED = os.path.join(os.environ.get("ZBL_MIXED_DATA", "/path/to/mixed_dataset/data"), "mixed_train.xyz")
RA, RB, POWER, B = 0.3, 1.5, 2.0, 0.5
OUT = Path(__file__).resolve().parents[1] / "results" / "selective_theta.json"


def cache(frames, calc, gate):
    out = []
    for at in frames:
        nov = gate.novelty(calc.get_descriptors(at))
        i, j, d, D = neighbor_list("ijdD", at, RB)
        at.calc = calc
        fv = np.asarray(at.get_forces()); fr = np.asarray(at.arrays["REF_forces"])
        out.append(dict(fv=fv, fr=fr, nov=nov, i=i, j=j, d=d, D=D, n=len(at)))
    return out


def corr_forces(rec, r_lo, r_hi, A):
    """F_corr[N,3] for a given θ from the cache (analytically, without MACE)."""
    F = np.zeros((rec["n"], 3))
    d = rec["d"]
    if len(d) == 0:
        return F
    rho = smoothstep(rec["nov"], r_lo, r_hi)
    w = np.maximum(rho[rec["i"]], rho[rec["j"]]) ** POWER
    bm = A * np.exp(-d / B); g = 1.0 - smoothstep(d, RA, RB)
    dVdr = w * (bm * (-1.0 / B) * g + bm * (-dsmoothstep(d, RA, RB)))
    np.add.at(F, rec["i"], (dVdr / d)[:, None] * rec["D"])
    return F


def fmae(cached, r_lo, r_hi, A):
    e = []
    for r in cached:
        fp = r["fv"] + corr_forces(r, r_lo, r_hi, A)
        e.append(np.abs(fp - r["fr"]))
    return np.concatenate([x.reshape(-1) for x in e]).mean()


def maxdf(cached, r_lo, r_hi, A):
    m = 0.0
    for r in cached:
        fc = corr_forces(r, r_lo, r_hi, A)
        m = max(m, np.abs(fc).max() if fc.size else 0.0)
    return m


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(model_paths=[VAN], device=dev, default_dtype="float32", head="mp_pbe_refit_add")
    gate = RNDGate(dev)
    keep = cache(read(PRE / "keep_train.xyz", index=":"), calc, gate)
    u200 = cache(read(PRE / "u200_train.xyz", index=":"), calc, gate)
    mpc = [a for a in read(MIXED, index=":") if a.info["REF_energy"] / len(a) < 50][1500:1800]
    mptrj = cache(mpc, calc, gate)
    base = u200 + mptrj
    print(f"calib: keep_train={len(keep)} u200_train={len(u200)} mptrj_calib={len(mptrj)}")

    # baseline (without correction)
    keep0 = fmae(keep, 1, 2, 0); base0 = fmae(base, 1, 2, 0)
    print(f"baseline F-MAE: keep={keep0:.3f}  base={base0:.4f} eV/Å\n")

    EPS = 0.01            # allowed increase of base F-MAE (risk-coverage budget)
    MAXDF_BASE = 1.0      # extra protection: max|ΔF| on the base ≤ 1 eV/Å
    print(f"{'r_lo':>6s} {'A':>6s} | {'keep MAE':>8s} {'Δkeep%':>7s} | {'base MAE':>8s} {'Δbase':>8s} {'maxΔF_base':>10s} {'OK':>4s}")
    print("-" * 78)
    best = None
    for r_lo in (0.02, 0.03, 0.05, 0.07, 0.1):
        r_hi = 10 * r_lo
        for A in (600, 800, 1200, 1600, 2400):
            km = fmae(keep, r_lo, r_hi, A)
            bm = fmae(base, r_lo, r_hi, A)
            mdf = maxdf(base, r_lo, r_hi, A)
            ok = (bm - base0) <= EPS and mdf <= MAXDF_BASE
            flag = "✓" if ok else ""
            print(f"{r_lo:6.3f} {A:6.0f} | {km:8.3f} {(km/keep0-1)*100:6.1f}% | {bm:8.4f} {bm-base0:+8.4f} {mdf:10.3f} {flag:>4s}")
            if ok and (best is None or km < best[3]):
                best = (r_lo, r_hi, A, km)
    print()
    if best is None:
        print("NO θ within budget — relax EPS"); return
    r_lo, r_hi, A, km = best
    OUT.write_text(json.dumps(dict(r_lo=r_lo, r_hi=r_hi, A=A, b=B, power=POWER, ra=RA, rb=RB,
                                   keep_train_mae=km, keep0=keep0, base0=base0), indent=2))
    print(f"BEST θ: r_lo={r_lo} r_hi={r_hi} A={A}  keep_train F-MAE {keep0:.2f}→{km:.2f} ({(km/keep0-1)*100:.1f}%)")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
