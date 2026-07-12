#!/usr/bin/env python3
"""ASE-MD baseline: the same calculator as in the LAMMPS driver, but MD is run by ASE.

For a fair ms/step comparison (LAMMPS fix external vs pure ASE).
Run: PYTHONPATH=pylibs deepmd-python ase_md_baseline.py --calc mace --data cu256.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.md.langevin import Langevin
from ase.units import fs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlip_fixext as drv


class A:
    model = None
    head = "omat_pbe"
    dtype = "float32"
    no_cueq = False
    anchor_mode = "pairphys"
    core_zbl = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", default="mace")
    ap.add_argument("--data", default="cu256.npz")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--no-cueq", action="store_true")
    args = ap.parse_args()

    z = np.load(args.data)
    atoms = Atoms("Cu" * len(z["atype"]), positions=z["coord"], cell=z["cell"], pbc=True)

    a = A()
    a.calc = args.calc
    a.no_cueq = args.no_cueq
    backend = (drv.MaceBackend if args.calc.startswith("mace") else drv.DpaBackend)(a)

    if args.calc.startswith("mace"):
        atoms.calc = backend.calc  # native ASE calculator
        dyn_atoms = atoms
    else:
        # DeepPot: wrapper via deepmd ASE calculator
        from deepmd.calculator import DP

        atoms.calc = DP(model=drv.DPA_MODEL)
        dyn_atoms = atoms

    dyn = Langevin(dyn_atoms, timestep=1.0 * fs, temperature_K=300, friction=0.01)
    dyn.run(5)  # warmup
    t0 = time.perf_counter()
    dyn.run(args.steps)
    wall = time.perf_counter() - t0
    print(f"ASE-MD {args.calc} {len(atoms)} atoms: {wall*1000/args.steps:.2f} ms/step")


if __name__ == "__main__":
    main()
