#!/usr/bin/env python3
"""xyz/extxyz -> LAMMPS data file. Prints the element order for --elems.

Usage: python xyz2data.py in.xyz out.data [index]
"""
import sys

from ase.io import read, write

src, dst = sys.argv[1], sys.argv[2]
idx = sys.argv[3] if len(sys.argv) > 3 else "0"
atoms = read(src, index=int(idx))
elems = sorted(set(atoms.get_chemical_symbols()))
write(dst, atoms, format="lammps-data", masses=True, specorder=elems)
print(f"{len(atoms)} atoms -> {dst}")
print(f"--elems {' '.join(elems)}")
