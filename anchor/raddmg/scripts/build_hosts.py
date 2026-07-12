#!/usr/bin/env python3
"""Build clean crystalline actinide-fluorite hosts (UO2/ThO2/NpO2) from scratch.

Fm-3m fluorite: cations on an FCC sublattice, anions on all 8 tetrahedral sites.
Optionally relax cell+positions with the `vanilla` calculator (the anchor does not
move the equilibrium), validate the space group with spglib, and write the relaxed
conventional cell + a manifest. Supercells are built on demand by the test drivers
via `.repeat`.

    python build_hosts.py --hosts UO2 ThO2 NpO2          # relax (needs GPU)
    python build_hosts.py --hosts UO2 --no-relax         # geometry only (CPU smoke)
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.io import write

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
HOSTS_DIR = ROOT / "raddmg" / "hosts"
LOG_DIR = ROOT / "raddmg" / "logs"

# fluorite AX2: (cation, anion, a0 Å) ; perovskite ABX3: (A, B, X, a0 Å)
FLUORITES = {
    "UO2":  ("U", "O", 5.47),
    "ThO2": ("Th", "O", 5.60),
    "NpO2": ("Np", "O", 5.43),
}
PEROVSKITES = {
    "SrTiO3": ("Sr", "Ti", "O", 3.905),
}
ALL_HOSTS = list(FLUORITES) + list(PEROVSKITES)


def fluorite_conventional(cation: str, anion: str, a: float) -> Atoms:
    cat = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
    an = list(itertools.product([.25, .75], repeat=3))
    return Atoms([cation] * 4 + [anion] * 8, scaled_positions=cat + an,
                 cell=[a, a, a], pbc=True)


def perovskite_conventional(A: str, B: str, X: str, a: float) -> Atoms:
    pos = [(0, 0, 0), (.5, .5, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
    return Atoms([A, B, X, X, X], scaled_positions=pos, cell=[a, a, a], pbc=True)


def conv_cell(host: str):
    """Return (conventional Atoms, expected space-group number, a0_literature)."""
    if host in FLUORITES:
        c, an, a = FLUORITES[host]
        return fluorite_conventional(c, an, a), 225, a
    A, B, X, a = PEROVSKITES[host]
    return perovskite_conventional(A, B, X, a), 221, a


def spacegroup_number(atoms: Atoms, symprec: float = 1e-2) -> int:
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return int(ds.number if hasattr(ds, "number") else ds["number"])


def relax_cell(atoms: Atoms, calc, logfile: str):
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=logfile)
    opt.run(fmax=0.02, steps=300)
    return atoms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hosts", nargs="+", default=ALL_HOSTS, choices=ALL_HOSTS)
    p.add_argument("--no-relax", action="store_true",
                   help="skip MLIP relaxation (geometry-only CPU smoke)")
    p.add_argument("--calc", default="vanilla",
                   help="calculator tag used for relaxation")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    HOSTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    calc = None
    if not args.no_relax:
        from common_calc import make_calculator
        calc = make_calculator(args.calc, device=args.device)

    mfile = HOSTS_DIR / "manifest.json"
    manifest = json.loads(mfile.read_text()) if mfile.exists() else {}
    for host in args.hosts:
        at, expected_sg, a0 = conv_cell(host)
        sg0 = spacegroup_number(at)
        e_per_atom = None
        if calc is not None:
            at = relax_cell(at, calc, str(LOG_DIR / f"relax_{host}.log"))
            e_per_atom = float(at.get_potential_energy() / len(at))
        sg = spacegroup_number(at)
        lengths = at.cell.lengths().tolist()
        out = HOSTS_DIR / f"{host}_conv.xyz"
        write(str(out), at)
        manifest[host] = dict(
            species=sorted(set(at.get_chemical_symbols())), a0_literature=a0,
            a_relaxed=lengths, a_mean=float(np.mean(lengths)),
            spacegroup_initial=sg0, spacegroup_final=sg, expected_sg=expected_sg,
            n_atoms=len(at), e_per_atom=e_per_atom,
            relaxed=(calc is not None), calc=args.calc if calc else None,
            file=str(out),
        )
        print(f"{host}: a={np.mean(lengths):.4f} Å  SG={sg}  "
              f"{'E/at=%.4f' % e_per_atom if e_per_atom is not None else '(no relax)'}",
              flush=True)
        if sg != expected_sg:
            print(f"  WARNING: {host} space group {sg} != {expected_sg}", flush=True)

    mfile.write_text(json.dumps(manifest, indent=1))
    print("→", mfile, flush=True)


if __name__ == "__main__":
    main()
