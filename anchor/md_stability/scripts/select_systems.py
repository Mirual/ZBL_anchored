#!/usr/bin/env python3
"""Selection of initial configs for MD: normal (baseline control), hot (perovskites → collapse at high T),
compressed (moderately compressed keep, distorted start but not collapsed).

Runnability criterion: dmin must not be extreme (<0.9 Å = already collapse) — otherwise everything blows up.
We prefer small cells (faster MD). Saved to systems/<set>_<i>.xyz."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list

PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
MPTRJ = os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz")
OUT = Path(__file__).resolve().parents[1] / "systems"


def dmin(at):
    d = neighbor_list("d", at, 3.0)
    return float(d.min()) if len(d) else 9.0


def main():
    OUT.mkdir(exist_ok=True)
    picked = {}

    # normal: small MPtrj structures (8–40 atoms), dmin>1.3 (normal bonds) — baseline control
    mp = read(MPTRJ, index="0:2000")
    cand = [a for a in mp if 8 <= len(a) <= 40 and dmin(a) > 1.3]
    cand.sort(key=len)
    normal = cand[:3]

    # hot: u200 pure perovskites, small, dmin>1.3 — run at high T to provoke collapse
    u = read(PRE / "u200_test.xyz", index=":")
    uc = [a for a in u if len(a) <= 60 and dmin(a) > 1.3]
    uc.sort(key=len)
    hot = uc[:2] if len(uc) >= 2 else uc + uc[:2 - len(uc)]

    # compressed: keep with dmin in [1.1, 1.6] (distorted but runnable), small
    kp = read(PRE / "keep_test.xyz", index=":")
    kc = [(a, dmin(a)) for a in kp if len(a) <= 120]
    kc = [a for a, dm in kc if 1.1 <= dm <= 1.6]
    kc.sort(key=len)
    compressed = kc[:5]

    for tag, frames in [("normal", normal), ("hot", hot), ("compressed", compressed)]:
        for i, at in enumerate(frames):
            f = OUT / f"{tag}_{i}.xyz"
            write(f, at)
            print(f"{f.name:16s} n={len(at):3d}  dmin={dmin(at):.2f} Å  {at.get_chemical_formula()[:30]}")
        picked[tag] = len(frames)
    print(f"\nTotal: {picked}")


if __name__ == "__main__":
    main()
