#!/usr/bin/env python3
"""SelectiveNet risk-coverage calibration of per-pair physics under the RND gate (port of MACE tune_pairphys.py).

corr_F(i,j) = λ · w_ij · dΔV_pair/dr,  w_ij = max(ρ_i,ρ_j)^power, ρ=smoothstep(novelty; r_lo,r_hi).
Calibration (r_lo, λ): min F-MAE(keep_train) s.t. ΔF-MAE(u200) ≤ ε, max|ΔF|_base ≤ 1.

Difference from the MACE version: the gate thresholds (r_lo, r_hi) are taken from the DISTRIBUTION of DPA novelty on
calibration atoms (p80/p90/p95, r_hi=p99.5), not from the fixed MACE grid {0.05,0.1} — the novelty scale for DPA
(dim 128) is different. Discrete ΔV values are cached to disk (results/dimer_dpa.pkl) for reuse.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpa_common import FOUNDATION, SPLITS, load_dp, compute_descriptors
from pair_physics import DimerCache
from gate import RNDGate, smoothstep

POWER = 2.0
RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "pairphys_theta.json"
DIMER_PKL = RES / "dimer_dpa.pkl"


def cache(frames, calc, gate, dc):
    """Per-frame: novelty, neighbour-list (i,j,d,D), dΔV/dr@λ1w1, vanilla & ref forces."""
    descs = compute_descriptors(frames)
    out = []
    for at, desc in zip(frames, descs):
        nov = gate.novelty(desc)
        Zs = np.unique(at.numbers)
        rmax = max(dc.kappa * (covalent_radii[a] + covalent_radii[b]) + dc.width for a in Zs for b in Zs)
        i, j, d, D = neighbor_list("ijdD", at, float(rmax))
        dV0dr = np.zeros(len(d)); Zn = at.numbers
        for key in {(min(Zn[a], Zn[b]), max(Zn[a], Zn[b])) for a, b in zip(i, j)}:
            e = dc.get(*key); zi, zj = key
            m = ((Zn[i] == zi) & (Zn[j] == zj)) | ((Zn[i] == zj) & (Zn[j] == zi))
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
    calc = load_dp(FOUNDATION)
    gate = RNDGate(dev); dc = DimerCache(calc, cache_path=DIMER_PKL)
    keep = cache(read(f"{SPLITS}/keep_train.xyz", index=":"), calc, gate, dc)
    base = cache(read(f"{SPLITS}/u200_train.xyz", index=":"), calc, gate, dc)
    dc.save()
    print(f"calib: keep={len(keep)} base(u200)={len(base)}  pairs cached={len(dc._cache)}")

    # gate thresholds from the distribution of DPA novelty on calibration atoms
    nov_all = np.concatenate([r["nov"] for r in keep + base])
    p80, p90, p95, p995 = np.percentile(nov_all, [80, 90, 95, 99.5])
    r_hi = float(p995)
    print(f"DPA novelty calib.: p80={p80:.4f} p90={p90:.4f} p95={p95:.4f} p99.5(r_hi)={r_hi:.4f}\n")

    keep0 = fmae(keep, 1, 2, 0); base0 = fmae(base, 1, 2, 0)
    print(f"baseline F-MAE: keep={keep0:.3f} base={base0:.4f}\n")

    EPS, MAXB = 0.01, 1.0
    print(f"{'r_lo':>8s} {'λ':>6s} | {'keep MAE':>8s} {'Δkeep%':>7s} | {'Δbase':>9s} {'maxΔF_b':>8s} {'OK':>3s}")
    print("-" * 66)
    best = None
    for r_lo in (float(p80), float(p90), float(p95)):
        for lam in (0.4, 0.8, 1.6, 3.0, 6.0, 12.0):
            km = fmae(keep, r_lo, r_hi, lam); bm = fmae(base, r_lo, r_hi, lam); mdf = maxdf(base, r_lo, r_hi, lam)
            ok = (bm - base0) <= EPS and mdf <= MAXB
            print(f"{r_lo:8.4f} {lam:6.2f} | {km:8.3f} {(km/keep0-1)*100:6.1f}% | {bm-base0:+9.4f} {mdf:8.3f} {'✓' if ok else '':>3s}")
            if ok and (best is None or km < best[3]):
                best = (r_lo, r_hi, lam, km)
    print()
    if best is None:
        print("NO θ within budget"); return
    r_lo, r_hi, lam, km = best
    OUT.write_text(json.dumps(dict(r_lo=r_lo, r_hi=r_hi, lam=lam, power=POWER,
                                   keep0=keep0, keep_mae=km), indent=2))
    print(f"BEST θ: r_lo={r_lo:.4f} λ={lam}  keep F-MAE {keep0:.2f}→{km:.2f} ({(km/keep0-1)*100:.1f}%)\nsaved {OUT}")


if __name__ == "__main__":
    main()
