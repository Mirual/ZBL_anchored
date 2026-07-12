#!/usr/bin/env python3
"""Precompute the DimerCache table of dimer curves to DISK (once per model → reuse everywhere).

The V_model-dimer(r) curve for a pair (Z_i,Z_j) depends ONLY on the model and the pair (not on structures/hyperparameters),
so it can be computed once and loaded at inference/calibration without recomputation.

Note: the table is specific to (model, ZBL state). For the pairphys deployment (built-in ZBL ON) — without
--disable-zbl. For the standalone core_zbl variant (ZBL OFF) — with --disable-zbl, a separate file.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
from ase.io import read
from ase.data import atomic_numbers
from mace.calculators import MACECalculator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_physics import DimerCache  # noqa: E402

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
USER = "Cl Cs F Fe Li Mg Mn N Np O P S Sr Tc Th Ti U Y Zr"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="path to the table (pickle)")
    p.add_argument("--data", default=None, help="extxyz — extract elements from it")
    p.add_argument("--extra", default=USER, help="extra elements (default — 19 user)")
    p.add_argument("--disable-zbl", action="store_true")
    args = p.parse_args()
    calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
    if args.disable_zbl:
        for m in calc.models:
            if hasattr(m, "pair_repulsion"):
                del m.pair_repulsion
    els = set(args.extra.split())
    if args.data:
        for a in read(args.data, index=":"):
            els.update(a.get_chemical_symbols())
    Zs = sorted({atomic_numbers[s] for s in els if s in atomic_numbers})
    pairs = [(zi, zj) for i, zi in enumerate(Zs) for zj in Zs[i:]]
    print(f"elements={len(Zs)}  pairs={len(pairs)}  zbl={'OFF' if args.disable_zbl else 'ON'}  → {args.out}", flush=True)
    dc = DimerCache(calc, cache_path=args.out)   # loads a partial table if present
    t0 = time.time()
    for k, (zi, zj) in enumerate(pairs):
        key = (min(zi, zj), max(zi, zj))
        if key in dc._cache:
            continue
        dc.get(zi, zj)
        if (k + 1) % 50 == 0:
            dc.save()  # periodically save (resume-friendly)
            print(f"  {k+1}/{len(pairs)}  cached={len(dc._cache)}  {time.time()-t0:.0f}s", flush=True)
    dc.save()
    print(f"DONE: {len(dc._cache)} pairs → {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
