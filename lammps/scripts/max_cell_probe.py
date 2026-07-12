#!/usr/bin/env python3
"""Maximum cell size on GPU: grow Cu-fcc until OOM + bisection.

Cu fcc density (~85 atoms/nm³) — a conservative reference for oxides and
metals. Measures peak memory and ms/step at each successful size.

Run: PYTHONPATH=pylibs deepmd-python max_cell_probe.py --calc mace
"""
import argparse
import gc
import json
import sys
import time
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
    anchor_cueq = False
    with_stress = False
    compile = False


def make_atoms(rep: int):
    from ase.build import bulk

    a = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((rep, rep, rep))
    a.rattle(stdev=0.05, seed=42)
    return a


def try_size(backend, rep: int):
    """(natoms, peak_MiB, ms) or None on OOM."""
    atoms = make_atoms(rep)
    n = len(atoms)
    try:
        backend.evaluate(atoms)  # warmup/allocation (OOM caught here)
        torch.cuda.reset_peak_memory_stats()
        atoms.positions += 1e-4  # otherwise the ASE cache returns a result without recomputing
        t0 = time.perf_counter()
        backend.evaluate(atoms)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(json.dumps({"natoms": n, "rep": rep, "peak_MiB": round(peak, 0),
                          "ms_per_eval": round(ms, 0)}), flush=True)
        return n, peak, ms
    except torch.cuda.OutOfMemoryError:
        print(json.dumps({"natoms": n, "rep": rep, "oom": True}), flush=True)
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", required=True)
    ap.add_argument("--anchor-cueq", action="store_true")
    ap.add_argument("--reps", type=int, nargs="+", default=None)
    args = ap.parse_args()

    a = A()
    a.calc = args.calc
    a.anchor_cueq = args.anchor_cueq
    backend = (drv.MaceBackend if args.calc.startswith("mace") else drv.DpaBackend)(a)

    reps = args.reps or ([8, 12, 17, 24, 34, 48] if args.calc.startswith("mace")
                         else [6, 8, 11, 15, 21])
    last_ok, first_fail = None, None
    for rep in reps:
        r = try_size(backend, rep)
        if r is None:
            first_fail = rep
            break
        last_ok = rep
    # bisection between last_ok and first_fail (by rep)
    while last_ok and first_fail and first_fail - last_ok > 1:
        mid = (last_ok + first_fail) // 2
        r = try_size(backend, mid)
        if r is None:
            first_fail = mid
        else:
            last_ok = mid
    if last_ok:
        print(json.dumps({"MAX": True, "calc": args.calc +
                          ("-acueq" if args.anchor_cueq else ""),
                          "rep": last_ok, "natoms": 4 * last_ok**3}), flush=True)


if __name__ == "__main__":
    main()
