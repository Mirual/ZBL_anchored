#!/usr/bin/env python3
"""DPA-3.1 in LAMMPS via a python-driven `fix external` + DeepPot (GPU).

Works with the frozen .pth AS IS (without border_op / re-freezing): on every
step the callback receives coordinates, DeepPot computes E/F/virial over the full
cell (single MPI rank → all atoms local → exactly the ASE path).

Modes:
  --mode validate : run 0, prints E and writes forces -> JSON (compare with ref_dpa.json)
  --mode bench    : NVE, N steps, ms/step
"""
import argparse
import json
import os
import time

import numpy as np
from deepmd.infer.deep_pot import DeepPot
from lammps import lammps

MODEL = os.environ.get("ZBL_DPA_MODEL", "/path/to/dpa-3.1-3m-ft.pth")
DATA = "data_cu256.lammps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "bench"], default="validate")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--elem", default="Cu")
    args = ap.parse_args()

    dp = DeepPot(args.model)
    type_idx = dp.get_type_map().index(args.elem)

    lmp = lammps(cmdargs=["-log", f"log.fixext_{args.mode}", "-screen", "none"])
    lmp.commands_string(f"""
units metal
boundary p p p
atom_style atomic
read_data {args.data}
mass 1 63.546
fix dpa all external pf/callback 1 1
fix_modify dpa energy yes virial yes
""")

    state = {"ncalls": 0, "last_f": None, "last_e": None}

    def callback(caller, ntimestep, nlocal, tag, x, f):
        # single rank: nlocal == natoms; sort by tag for a stable order
        order = np.argsort(tag)
        coord = x[order].reshape(1, -1)
        box = lmp.extract_box()
        (xlo, ylo, zlo), (xhi, yhi, zhi) = box[0], box[1]
        xy, yz, xz = box[2], box[3], box[4]
        cell = np.array(
            [[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]]
        ).reshape(1, -1)
        atype = [type_idx] * nlocal
        e, force, virial = dp.eval(coord, cell, atype)
        fr = np.asarray(force).reshape(-1, 3)
        # return forces in the order of the LAMMPS local indices
        f[order] = fr
        lmp.fix_external_set_energy_global("dpa", float(e[0][0]))
        # deepmd virial: 9-component (sum of r⊗F); LAMMPS expects 6 (xx,yy,zz,xy,xz,yz)
        v = np.asarray(virial).reshape(3, 3)
        v6 = [v[0, 0], v[1, 1], v[2, 2], v[0, 1], v[0, 2], v[1, 2]]
        lmp.fix_external_set_virial_global("dpa", v6)
        state["ncalls"] += 1
        state["last_e"] = float(e[0][0])
        state["last_f"] = fr.copy()  # fr already in tag order (1..N = order of the data file)

    lmp.set_fix_external_callback("dpa", callback, lmp)

    if args.mode == "validate":
        lmp.command("run 0 post no")
        # last_f: forces in tag order (1..N) = order of the data file = order of the npz
        fr_tag = state["last_f"]
        out = {"energy_eV": state["last_e"], "forces": fr_tag.tolist()}
        with open("lmp_fixext_run0.json", "w") as fh:
            json.dump(out, fh)
        pe = lmp.get_thermo("pe")
        print(f"callback E = {state['last_e']:.8f} eV; thermo pe = {pe:.8f} eV")
    else:
        lmp.commands_string("""
velocity all create 300.0 4928459 mom yes rot yes dist gaussian
fix 1 all nve
timestep 0.001
thermo 50
""")
        lmp.command("run 5 post no")  # warm-up
        t0 = time.perf_counter()
        lmp.command(f"run {args.steps} post no")
        dt = time.perf_counter() - t0
        print(f"BENCH: {args.steps} steps, {dt*1000/args.steps:.2f} ms/step, calls={state['ncalls']}")
        print(f"final pe = {lmp.get_thermo('pe'):.6f} eV, T = {lmp.get_thermo('temp'):.1f} K")

    lmp.close()


if __name__ == "__main__":
    main()
