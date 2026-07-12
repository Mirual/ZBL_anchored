#!/usr/bin/env python3
"""Sanity: 3-way (vanilla/anchor/FT) on the EXTREME keep set (where the anchor acts),
+ diagnose anchor activity vs the milder mixed_test 'distorted' group.

Answers: is anchor≡vanilla on mixed_test a bug, or just because mixed_test frames are too
mild to fire the gate? Compares keep_test (extreme compression) to mixed_test distorted.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

ROOT = Path(os.environ.get("ZBL_IAML_WS", ""))  # TODO: set for your machine (external workspace)
if os.environ.get("ZBL_IAML_WS"):
    sys.path.insert(0, str(ROOT / "idea_uncertainty_gated_physics_anchor" / "raddmg" / "scripts"))
from common_calc import make_calculator  # noqa: E402

KEEP = ROOT / "vasp_tier1/collected/preflight/splits/keep_test.xyz"
MIXED = ROOT / "mixed_dataset_clean/data/mixed_test.xyz"
FT = ROOT / "mace_mh_mix_clean/results/mace_mh0_mix_clean_F10E1/mace_mh0_mix_clean_F10E1.model"


def get_ref(at):
    for k in ("REF_forces", "forces"):
        if k in at.arrays:
            return np.asarray(at.arrays[k], float)
    return None


def r2(pred, ref):
    p = np.concatenate([x.reshape(-1) for x in pred]); r = np.concatenate([x.reshape(-1) for x in ref])
    return float(1 - ((p - r) ** 2).sum() / ((r - r.mean()) ** 2).sum())


def mae(pred, ref):
    p = np.concatenate([x.reshape(-1) for x in pred]); r = np.concatenate([x.reshape(-1) for x in ref])
    return float(np.mean(np.abs(p - r)))


def fmin(at):
    d = neighbor_list("d", at, 5.0)
    return float(d.min()) if len(d) else 9.0


def main():
    van = make_calculator("vanilla")
    pp = make_calculator("vanilla_pairphys")
    ft = MACECalculator(model_paths=[str(FT)], device="cuda", default_dtype="float32", head="Default")

    keep = read(str(KEEP), index=":")
    print(f"keep_test: {len(keep)} frames; min_dist range "
          f"{min(fmin(a) for a in keep):.2f}–{max(fmin(a) for a in keep):.2f} Å")

    pools = {m: [] for m in ("vanilla", "anchor", "ft")}; ref = []
    dF_all = []   # |F_anchor - F_vanilla| per atom (anchor activity)
    for at in keep:
        Fref = get_ref(at)
        if Fref is None:
            continue
        Fv = forces(van, at); Fp = forces(pp, at); Ff = forces(ft, at)
        ref.append(Fref); pools["vanilla"].append(Fv); pools["anchor"].append(Fp); pools["ft"].append(Ff)
        dF_all.append(np.linalg.norm(Fp - Fv, axis=1))
    dF = np.concatenate(dF_all)
    print(f"\nanchor ACTIVITY on keep: atoms={len(dF)}  touched(|ΔF|>0.05)={int((dF>0.05).sum())} "
          f"({(dF>0.05).mean():.1%})  max|ΔF|={dF.max():.1f} eV/Å")
    print("\n=== 3-way on keep_test (forces vs DFT) ===")
    for m in ("vanilla", "anchor", "ft"):
        print(f"  {m:8s}: F R² {r2(pools[m], ref):.3f}   F-MAE {mae(pools[m], ref):.3f}")

    # contrast: anchor activity on mixed_test distorted (<1.5 Å) frames
    mt = read(str(MIXED), index=":")
    dist = [a for a in mt if fmin(a) < 1.5]
    print(f"\nmixed_test 'distorted'(<1.5Å): {len(dist)} frames; min_dist range "
          f"{min(fmin(a) for a in dist):.2f}–{max(fmin(a) for a in dist):.2f} Å")
    dF2 = []
    for at in dist[:200]:
        Fv = forces(van, at); Fp = forces(pp, at)
        dF2.append(np.linalg.norm(Fp - Fv, axis=1))
    dF2 = np.concatenate(dF2)
    print(f"anchor ACTIVITY on mixed_test-distorted: atoms={len(dF2)}  touched(|ΔF|>0.05)="
          f"{int((dF2>0.05).sum())} ({(dF2>0.05).mean():.2%})  max|ΔF|={dF2.max():.2f} eV/Å")


def forces(calc, at):
    a = at.copy(); a.calc = calc
    return np.asarray(a.get_forces(), float)


if __name__ == "__main__":
    main()
