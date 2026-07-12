#!/usr/bin/env python3
"""Reference for anchor validation: direct call of the same backends (without LAMMPS)
on the geometry FROM the LAMMPS data file (the same rotated triclinic frame that
LAMMPS will see). Plus the magnitude of the anchor correction (the gate should fire).

Run: PYTHONPATH=pylibs deepmd-python ref_anchor.py <data> <calc> <out.json> <elem1> [elem2 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mlip_fixext as drv


class A:  # minimal args object for the backends
    model = None
    head = "omat_pbe"
    dtype = "float32"
    no_cueq = False
    anchor_mode = "pairphys"
    core_zbl = False


def main():
    data, calcname, out = sys.argv[1], sys.argv[2], sys.argv[3]
    elems = sys.argv[4:]
    zmap = {i + 1: atomic_numbers[s] for i, s in enumerate(elems)}
    atoms = read(data, format="lammps-data", Z_of_type=zmap, style="atomic")
    atoms.pbc = True
    args = A()
    args.calc = calcname
    backend = (drv.MaceBackend if calcname.startswith("mace") else drv.DpaBackend)(args)
    e, f, v6 = backend.evaluate(atoms)

    res = {"calc": calcname, "energy_eV": e, "forces": np.asarray(f).tolist(),
           "natoms": len(atoms)}

    if calcname == "mace-anchor":
        base = backend.calc.mace  # the same MACE-MH-0 inside
        probe = atoms.copy()
        probe.calc = base
        e0 = float(probe.get_potential_energy())
        res["E_vanilla_eV"] = e0
        res["E_corr_eV"] = e - e0
    elif calcname == "dpa-anchor":
        backend.anchor = False
        e0, _, _ = backend.evaluate(atoms)
        backend.anchor = True
        res["E_vanilla_eV"] = e0
        res["E_corr_eV"] = e - e0

    msg = f"{calcname}: E = {e:.6f} eV"
    if "E_corr_eV" in res:
        msg += f" (anchor correction {res['E_corr_eV']:+.4f} eV)"
    print(msg)
    Path(out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
