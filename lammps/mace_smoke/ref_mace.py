#!/usr/bin/env python3
"""ASE reference E/F for cu256: MACE-MH-1, fp32, e3nn (as in the mliap wrapper).

Run in the mace-cueq env.
"""
import json
import os
import sys

import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator

MODEL = os.environ.get("ZBL_MACE_MH1", "/path/to/mace-mh-1.model")
HEAD = "omat_pbe"

z = np.load("../dpa_smoke/cu256.npz")
atoms = Atoms("Cu256", positions=z["coord"], cell=z["cell"], pbc=True)
calc = MACECalculator(
    model_paths=[MODEL], device="cuda", default_dtype="float32", head=HEAD
)
atoms.calc = calc
e = float(atoms.get_potential_energy())
f = atoms.get_forces()
json.dump(
    {"energy_eV": e, "forces": f.tolist(), "head": HEAD},
    open("ref_mace_mh1_fp32.json", "w"),
)
print(f"E = {e:.6f} eV; Fmax = {np.abs(f).max():.4f} eV/A")
