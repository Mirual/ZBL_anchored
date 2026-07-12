#!/usr/bin/env python3
"""GPU memory profile of calculators in the MD loop (vanilla vs anchor).

Measures steady-state peak torch.cuda.max_memory_allocated over 15 evaluate()
calls with light position jitter (as in MD), after 3 warmup ones.

Run: PYTHONPATH=pylibs deepmd-python gpu_mem_profile.py --calc mace --data cu256.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlip_fixext as drv


class A:
    model = None
    head = "omat_pbe"
    dtype = "float32"
    no_cueq = False
    anchor_mode = "pairphys"
    core_zbl = False
    anchor_lite = False
    with_stress = False
    compile = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", required=True)
    ap.add_argument("--data", required=True, help="npz with coord/cell (Cu)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cueq", action="store_true")
    ap.add_argument("--anchor-lite", action="store_true")
    ap.add_argument("--anchor-cueq", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--head", default="omat_pbe")
    args = ap.parse_args()

    from ase import Atoms

    z = np.load(args.data)
    n = len(z["coord"])
    atoms = Atoms(f"Cu{n}", positions=z["coord"], cell=z["cell"], pbc=True)

    a = A()
    a.calc = args.calc
    a.model = args.model
    a.head = args.head
    a.no_cueq = args.no_cueq
    a.anchor_lite = args.anchor_lite
    a.anchor_cueq = args.anchor_cueq
    backend = (drv.MaceBackend if args.calc.startswith("mace") else drv.DpaBackend)(a)

    rng = np.random.default_rng(0)
    for _ in range(3):  # warmup
        atoms.positions += rng.normal(0, 1e-3, atoms.positions.shape)
        backend.evaluate(atoms)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(15):
        atoms.positions += rng.normal(0, 1e-3, atoms.positions.shape)
        backend.evaluate(atoms)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 2**20
    resid = torch.cuda.memory_allocated() / 2**20
    tag = args.calc + ("-nocueq" if args.no_cueq else "") + \
        ("-lite" if args.anchor_lite else "") + \
        ("-acueq" if args.anchor_cueq else "")
    res = {"calc": tag, "natoms": n,
           "peak_MiB": round(peak, 1), "resident_MiB": round(resid, 1)}
    print(json.dumps(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
