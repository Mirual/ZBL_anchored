#!/usr/bin/env python3
"""Compare LAMMPS run-0 vs DeepPot: dE and max|dF|. Without ase (deepmd env)."""
import json
import re
import sys

import numpy as np

ref = json.load(open("ref_dpa.json"))
f_ref = np.asarray(ref["forces"])

# forces from the dump (sort id already applied)
rows = []
with open("forces_run0.dump") as fh:
    lines = fh.readlines()
i = lines.index("ITEM: ATOMS id fx fy fz\n") + 1
for ln in lines[i:]:
    p = ln.split()
    rows.append([float(p[1]), float(p[2]), float(p[3])])
f_lmp = np.asarray(rows)

# energy from the PE_FULL log line
log = open("log.lammps").read()
m = re.search(r"PE_FULL: (-?\d+\.\d+)", log)
e_lmp = float(m.group(1))

dE = e_lmp - ref["energy_eV"]
dF = np.abs(f_lmp - f_ref)
res = {
    "E_lammps_eV": e_lmp,
    "E_deeppot_eV": ref["energy_eV"],
    "dE_eV": dE,
    "dE_meV_per_atom": dE / len(f_ref) * 1000,
    "maxdF_eV_A": float(dF.max()),
    "rmsdF_eV_A": float(np.sqrt((dF**2).mean())),
}
print(json.dumps(res, indent=2))
ok = abs(res["dE_meV_per_atom"]) < 1.0 and res["maxdF_eV_A"] < 1e-3
print("VERDICT:", "LOSSLESS-OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
