#!/usr/bin/env python3
"""Calibration of the global Born–Mayer anchor against REAL forces and MPtrj-preservation check.

1. Computes vanilla forces on keep (once) + pair geometry.
2. Grid over (A, b): F_total = F_vanilla + F_corr(A,b); metric — F MAE vs F_ref on keep.
3. Best A,b. Then checks that on MPtrj (normal bonds) the correction ≈ 0 → base intact.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_predict import pair_correction  # noqa: E402

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
RA, RB = 0.3, 1.5


def vanilla_forces(frames, calc):
    out = []
    for at in frames:
        at.calc = calc
        out.append(np.asarray(at.get_forces()))
    return out


def fmae(frames, Fv, A, b):
    num = den = 0.0
    for at, fv in zip(frames, Fv):
        fr = at.arrays["REF_forces"]
        _, fc = pair_correction(at, A, b, RA, RB)
        d = np.abs((fv + fc) - fr)
        num += d.sum(); den += d.size
    return num / den


def main():
    calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32",
                          head="mp_pbe_refit_add")
    keep = read(PRE / "splits" / "keep_test.xyz", index=":")
    Fv = vanilla_forces(keep, calc)
    base = fmae(keep, Fv, 0.0, 1.0)
    print(f"keep_test n={len(keep)}  vanilla F MAE = {base:.3f} eV/Å")
    print("\ngrid (A eV, b Å) → keep F MAE:")
    best = (base, 0.0, 1.0)
    for A in (20, 50, 100, 200, 400, 800):
        for b in (0.2, 0.3, 0.4, 0.5):
            v = fmae(keep, Fv, A, b)
            mark = ""
            if v < best[0]:
                best = (v, A, b); mark = "  <-- best"
            print(f"  A={A:4d} b={b:.1f}: {v:8.3f}{mark}")
    print(f"\nBEST: A={best[1]} b={best[2]} → keep F MAE {best[0]:.3f} (vanilla {base:.3f}, "
          f"improvement {(1-best[0]/base)*100:.0f}%)")

    # MPtrj-preservation check: correction on normal structures
    mp = read(os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), index="0:300")
    A, b = best[1], best[2]
    max_corr = 0.0; n_touched = 0
    for at in mp:
        _, fc = pair_correction(at, A, b, RA, RB)
        m = np.abs(fc).max()
        max_corr = max(max_corr, m)
        if m > 1e-3:
            n_touched += 1
    print(f"\nMPtrj(300): correction touches {n_touched}/300 structures, max |F_corr| = {max_corr:.4f} eV/Å")
    print("(if few and small → base barely changes; if many → r_b catches normal bonds)")


if __name__ == "__main__":
    main()
