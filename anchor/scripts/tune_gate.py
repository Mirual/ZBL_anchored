#!/usr/bin/env python3
"""Sweep (threshold ρ, A) with cached vanilla forces and RAW kNN distance.
Goal: find a config where MPtrj is preserved (F R²~vanilla) AND keep is better.
Diagnostic: at which threshold problematic MPtrj atoms are cut off, but useful keep atoms are not.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rho_anchor_predict import Rho, pair_corr_gated  # noqa: E402
from anchor_predict import smoothstep  # noqa: E402

VAN = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
RA, RB = 0.3, 1.5


def cache(frames, calc, rho):
    out = []
    for at in frames:
        raw = rho.raw_dist(np.asarray(calc.get_descriptors(at)))   # raw kNN dist
        at.calc = calc
        out.append((at, np.asarray(at.get_forces()), np.asarray(at.arrays["REF_forces"]),
                    raw, float(at.get_potential_energy()), float(at.info["REF_energy"]), len(at)))
    return out


def metrics(cached, A, b, power, r_lo, r_hi):
    Fr, Fp, dE_pa, Er, Ep = [], [], [], [], []
    for at, fv, fr, raw, ev, er, nat in cached:
        rho_i = smoothstep(raw, r_lo, r_hi)
        ec, fc = pair_corr_gated(at, rho_i, A, b, RA, RB, power)
        Fr.append(fr.reshape(-1)); Fp.append((fv + fc).reshape(-1))
        dE_pa.append(abs((ev + ec) - er) / nat); Er.append(er); Ep.append(ev + ec)
    Fr = np.concatenate(Fr); Fp = np.concatenate(Fp)
    fss = ((Fr - Fp) ** 2).sum(); fst = ((Fr - Fr.mean()) ** 2).sum()
    Er = np.array(Er); Ep = np.array(Ep)
    ess = ((Er - Ep) ** 2).sum(); est = ((Er - Er.mean()) ** 2).sum()
    return (np.abs(Fp - Fr).mean(), 1 - fss / fst,            # F MAE, F R²
            float(np.mean(dE_pa)), 1 - ess / est)              # E MAE/atom (eV), E R²


def main():
    calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
    rho = Rho()
    keep = cache(read(PRE / "splits" / "keep_test.xyz", index=":"), calc, rho)
    mp = cache(read(os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), index="0:500"), calc, rho)
    # diagnostic: distribution of raw dist keep vs MPtrj
    rk = np.concatenate([c[3] for c in keep]); rm = np.concatenate([c[3] for c in mp])
    print(f"raw kNN-dist: keep p50={np.percentile(rk,50):.2f} p90={np.percentile(rk,90):.2f} | "
          f"MPtrj p50={np.percentile(rm,50):.2f} p90={np.percentile(rm,90):.2f} p99={np.percentile(rm,99):.2f} max={rm.max():.2f}")
    kb = metrics(keep, 0, 1, 1, 1, 2); mb = metrics(mp, 0, 1, 1, 1, 2)
    print(f"VANILLA keep: F MAE={kb[0]:.2f} F R²={kb[1]:.3f} | E MAE={kb[2]:.2f} eV/at E R²={kb[3]:.3f}  (keep E=poison)")
    print(f"VANILLA MPtrj: F MAE={mb[0]:.3f} F R²={mb[1]:.3f} | E MAE={mb[2]*1000:.1f} meV/at E R²={mb[3]:.3f}\n")
    print(f"{'r_lo':>5s} {'A':>6s} | {'keep F MAE':>10s} {'keep FR²':>8s} {'keep E MAE':>10s} | "
          f"{'MP F MAE':>8s} {'MP FR²':>7s} {'MP E MAE':>9s} {'MP ER²':>7s}")
    print("-" * 92)
    for r_lo in (0.6, 1.5, 3.0):
        for A in (800, 1600):
            k = metrics(keep, A, 0.5, 2, r_lo, r_lo + 1.0)
            m = metrics(mp, A, 0.5, 2, r_lo, r_lo + 1.0)
            flag = "  <-- base intact" if (m[1] > 0.95 and m[3] > 0.99) else ""
            print(f"{r_lo:5.1f} {A:6.0f} | {k[0]:10.2f} {k[1]:8.3f} {k[2]:9.1f}eV | "
                  f"{m[0]:8.3f} {m[1]:7.3f} {m[2]*1000:7.1f}m {m[3]:7.3f}{flag}")


if __name__ == "__main__":
    main()
