#!/usr/bin/env python3
"""Layer 1 predict: vanilla MACE-MH-0 + per-pair dimer-residual correction (pair_physics).
NO ρ-gate — selectivity comes from the residual itself (self-vanishes at equilibrium) + κ-cutoff.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_physics import DimerCache, pair_correction

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kappa", type=float, default=0.90)
    p.add_argument("--width", type=float, default=0.40)
    args = p.parse_args()
    calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
    cache = DimerCache(calc, kappa=args.kappa, width=args.width)

    frames = read(args.data, index=":", format="extxyz")
    recs, nsk = [], 0
    for k, at in enumerate(frames):
        er = float(at.info.get("REF_energy", at.info.get("energy", 0.0)))
        fr = at.arrays.get("REF_forces")
        at.calc = calc
        try:
            ev = float(at.get_potential_energy()); fv = np.asarray(at.get_forces())
            ec, fc = pair_correction(at, cache)
            ep = ev + ec; fp = (fv + fc).tolist()
        except Exception:
            nsk += 1; ep, fp = None, None
        recs.append({"idx": k, "n_atoms": len(at), "E_ref": er, "E_pred": ep,
                     "F_ref": fr.tolist() if fr is not None else None, "F_pred": fp})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"frames": recs}, indent=2))
    print(f"{len(frames)} frames skipped {nsk} → {args.out}  (pairs cached: {len(cache._cache)})")


if __name__ == "__main__":
    main()
