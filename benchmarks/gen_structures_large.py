#!/usr/bin/env python3
"""Generate larger crystalline supercells for the cuEquivariance speedup-vs-size sweep.

cueq's fused kernels only pay off once the tensor-product / symmetric-contraction
work dominates the per-call overhead — i.e. on big cells (and/or in fp32). The
default bench cells (~16-27 atoms) are far too small to show it (measured ~1.0x).
This builds a deterministic size sweep, written byte-identical to extxyz so the
OFF (e3nn) and ON (cueq) runs use exactly the same geometry.

    python gen_structures_large.py [out.extxyz]
"""
import sys

import numpy as np
from ase.build import bulk
from ase.io import write


def build_large(seed: int = 0):
    """Deterministic supercells spanning ~27 -> ~1331 atoms (primitive cells)."""
    specs = [
        ("Cu", dict(crystalstructure="fcc", a=3.61), (3, 3, 3)),        # 27
        ("Cu", dict(crystalstructure="fcc", a=3.61), (5, 5, 5)),        # 125
        ("Cu", dict(crystalstructure="fcc", a=3.61), (7, 7, 7)),        # 343
        ("Cu", dict(crystalstructure="fcc", a=3.61), (9, 9, 9)),        # 729
        ("Cu", dict(crystalstructure="fcc", a=3.61), (11, 11, 11)),     # 1331
        ("Si", dict(crystalstructure="diamond", a=5.43), (8, 8, 8)),    # 1024
        ("MgO", dict(crystalstructure="rocksalt", a=4.21), (8, 8, 8)),  # 1024
    ]
    rng = np.random.default_rng(seed)
    atoms_list = []
    for name, kw, rep in specs:
        atoms = bulk(name, **kw).repeat(rep)
        # small seeded rattle so forces are nonzero but geometry stays in-distribution
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        atoms.info["label"] = f"{name}_{rep[0]}{rep[1]}{rep[2]}_{len(atoms)}"
        atoms_list.append(atoms)
    return atoms_list


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "structures_large.extxyz"
    atoms_list = build_large()
    write(path, atoms_list)
    for atoms in atoms_list:
        print(f"{atoms.info['label']:>16}  natoms={len(atoms)}")
    print(f"wrote {path}")
