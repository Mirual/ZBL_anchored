#!/usr/bin/env python3
"""#2: per-pair physics as the FORM of the correction under the RND gate + global scale λ.

corr_pair(i,j) = λ · w_ij · ΔV_pair(Z_i,Z_j, r_ij),  w_ij = max(ρ_i,ρ_j)^power, ρ=smoothstep(novelty).
ΔV_pair — per-pair ZBL residual (DimerCache, pair_physics). λ compensates for many-body screening
(an isolated dimer overestimates — see Layer-1). The RND gate keeps the magnitude in check (Layer-1 without the gate blew up).

SelectiveNet risk-coverage calibration (r_lo, λ) on disjoint train/calib splits:
    min F-MAE(keep_train)  s.t.  ΔF-MAE(base) ≤ ε,  max|ΔF|_base ≤ 1.
We cache per-pair (d, D, dV0, dV0dr) at λ=1,w=1 → analytical grid (r_lo, λ).
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
from anchor_predict import smoothstep
from pair_physics import DimerCache
from rnd_anchor_predict import RNDGate

VAN = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight")) / "splits"
MIXED = os.path.join(os.environ.get("ZBL_MIXED_DATA", "/path/to/mixed_dataset/data"), "mixed_train.xyz")
POWER = 2.0
OUT = Path(__file__).resolve().parents[1] / "results" / "pairphys_theta.json"


def cache(frames, calc, gate, dc):
    out = []
    for at in frames:
        nov = gate.novelty(calc.get_descriptors(at))
        Zs = np.unique(at.numbers)
        from ase.data import covalent_radii
        rmax = max(dc.kappa * (covalent_radii[a] + covalent_radii[b]) + dc.width for a in Zs for b in Zs)
        i, j, d, D = neighbor_list("ijdD", at, float(rmax))
        dV0 = np.zeros(len(d)); dV0dr = np.zeros(len(d))
        Zn = at.numbers
        for key in {(min(Zn[a], Zn[b]), max(Zn[a], Zn[b])) for a, b in zip(i, j)}:
            e = dc.get(*key); zi, zj = key
            m = ((Zn[i] == zi) & (Zn[j] == zj)) | ((Zn[i] == zj) & (Zn[j] == zi))
            dV0[m] = np.interp(d[m], e["r"], e["dV"], left=e["dV"][0], right=0.0)
            dV0dr[m] = np.interp(d[m], e["r"], e["dVdr"], left=e["dVdr"][0], right=0.0)
        at.calc = calc
        fv = np.asarray(at.get_forces()); fr = np.asarray(at.arrays["REF_forces"])
        out.append(dict(fv=fv, fr=fr, nov=nov, i=i, j=j, d=d, D=D, dV0dr=dV0dr, n=len(at)))
    return out


def corr_F(rec, r_lo, r_hi, lam):
    F = np.zeros((rec["n"], 3))
    if len(rec["d"]) == 0:
        return F
    rho = smoothstep(rec["nov"], r_lo, r_hi)
    w = np.maximum(rho[rec["i"]], rho[rec["j"]]) ** POWER
    dVdr = lam * w * rec["dV0dr"]
    np.add.at(F, rec["i"], (dVdr / rec["d"])[:, None] * rec["D"])
    return F


def fmae(C, r_lo, r_hi, lam):
    return np.concatenate([np.abs(r["fv"] + corr_F(r, r_lo, r_hi, lam) - r["fr"]).reshape(-1) for r in C]).mean()


def maxdf(C, r_lo, r_hi, lam):
    m = 0.0
    for r in C:
        fc = corr_F(r, r_lo, r_hi, lam); m = max(m, np.abs(fc).max() if fc.size else 0.0)
    return m


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(model_paths=[VAN], device=dev, default_dtype="float32", head="mp_pbe_refit_add")
    gate = RNDGate(dev); dc = DimerCache(calc)
    keep = cache(read(PRE / "keep_train.xyz", index=":"), calc, gate, dc)
    # base-coverage = u200_train (distorted-but-clean — binding constraint). MPtrj is NOT needed:
    # the RND gate at r_lo>=0.05 gives ρ=0 on 100% of MPtrj atoms (verified with --check) → the correction ≡0 there.
    base = cache(read(PRE / "u200_train.xyz", index=":"), calc, gate, dc)
    print(f"calib: keep={len(keep)} base(u200)={len(base)}  pairs cached={len(dc._cache)}")
    keep0 = fmae(keep, 1, 2, 0); base0 = fmae(base, 1, 2, 0)
    print(f"baseline F-MAE: keep={keep0:.3f} base={base0:.4f}\n")

    EPS, MAXB = 0.01, 1.0
    print(f"{'r_lo':>6s} {'λ':>6s} | {'keep MAE':>8s} {'Δkeep%':>7s} | {'Δbase':>9s} {'maxΔF_b':>8s} {'OK':>3s}")
    print("-" * 64)
    best = None
    for r_lo in (0.05, 0.1):
        r_hi = 10 * r_lo
        for lam in (0.4, 0.8, 1.6, 3.0, 6.0, 12.0):
            km = fmae(keep, r_lo, r_hi, lam); bm = fmae(base, r_lo, r_hi, lam); mdf = maxdf(base, r_lo, r_hi, lam)
            ok = (bm - base0) <= EPS and mdf <= MAXB
            print(f"{r_lo:6.3f} {lam:6.2f} | {km:8.3f} {(km/keep0-1)*100:6.1f}% | {bm-base0:+9.4f} {mdf:8.3f} {'✓' if ok else '':>3s}")
            if ok and (best is None or km < best[3]):
                best = (r_lo, r_hi, lam, km)
    print()
    if best is None:
        print("NO θ within budget"); return
    r_lo, r_hi, lam, km = best
    OUT.write_text(json.dumps(dict(r_lo=r_lo, r_hi=r_hi, lam=lam, power=POWER,
                                   keep0=keep0, keep_mae=km), indent=2))
    print(f"BEST θ: r_lo={r_lo} λ={lam}  keep F-MAE {keep0:.2f}→{km:.2f} ({(km/keep0-1)*100:.1f}%)\nsaved {OUT}")


if __name__ == "__main__":
    main()
