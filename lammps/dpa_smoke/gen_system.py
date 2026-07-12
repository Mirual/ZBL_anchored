#!/usr/bin/env python3
"""Test system for LAMMPS smoke tests: Cu fcc 4x4x4 (256 atoms) with rattle.

Writes data_cu256.lammps (units metal, atom_style atomic) + cu256.npz
(coord/cell/atype) for the python references. Run in an env with ase (mace-cueq).
"""
import numpy as np
from ase.build import bulk
from ase.io import write

OUT_DATA = "data_cu256.lammps"
OUT_NPZ = "cu256.npz"
RATTLE_STD = 0.05  # A — so that forces are non-zero
SEED = 42

atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((4, 4, 4))
atoms.rattle(stdev=RATTLE_STD, seed=SEED)
assert len(atoms) == 256

write(OUT_DATA, atoms, format="lammps-data", masses=True, specorder=["Cu"])
np.savez(
    OUT_NPZ,
    coord=atoms.get_positions(),
    cell=np.asarray(atoms.cell),
    atype=np.zeros(len(atoms), dtype=int),  # single type: Cu
)
print(f"OK: {len(atoms)} atoms -> {OUT_DATA}, {OUT_NPZ}")
