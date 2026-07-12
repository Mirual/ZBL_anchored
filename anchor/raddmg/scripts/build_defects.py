#!/usr/bin/env python3
"""Generate point-defect cells from a clean host (CPU-only, structure-agnostic).

For each requested species emits a vacancy and a Frenkel pair; plus one antisite swap;
plus the perfect reference. The Frenkel interstitial is placed in the DEEPEST void that
is ≥ min_sep from the created vacancy, so vacancy and interstitial do not recombine on
relaxation (this fixes the small-cell O_frenkel≈0 artifact). Works for fluorite and
perovskite alike (no hard-coded interstitial sites).

  mu-free (atom-conserving, directly comparable to DFT): Frenkel pairs, antisite
  mu-dependent (raw ΔE only): isolated vacancies

    python build_defects.py --host UO2    --nrep 3 --species U O
    python build_defects.py --host SrTiO3 --nrep 3 --species Sr Ti O
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read, write

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
HOSTS_DIR = ROOT / "raddmg" / "hosts"
DEFECTS_DIR = ROOT / "raddmg" / "defects"


def central_atom(atoms, species: str) -> int:
    center = atoms.cell[:].sum(axis=0) / 2.0
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == species]
    if not idx:
        raise ValueError(f"no {species} atoms in cell")
    d = np.linalg.norm(atoms.get_positions()[idx] - center, axis=1)
    return idx[int(np.argmin(d))]


def find_void(atoms, vac_pos, min_sep: float = 3.0, ngrid: int = 12):
    """Deepest interstitial void (max min-distance to atoms) that is ≥ min_sep from
    the vacancy site. Vectorised grid search in fractional coordinates."""
    cell = atoms.cell[:]
    fr = np.linspace(0.0, 1.0, ngrid, endpoint=False)
    grid = np.array(list(itertools.product(fr, repeat=3)))
    carts = grid @ cell
    _, D = get_distances(carts, atoms.get_positions(), cell=cell, pbc=True)   # (G, N)
    mind = D.min(axis=1)
    _, Dv = get_distances(carts, np.array([vac_pos]), cell=cell, pbc=True)    # (G, 1)
    dvac = Dv[:, 0]
    ok = dvac >= min_sep
    if not ok.any():
        ok = dvac >= 0.5 * min_sep
    cand = np.where(ok)[0]
    best = cand[int(np.argmax(mind[cand]))]
    return carts[best], float(mind[best])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--nrep", type=int, default=3)
    p.add_argument("--species", nargs="+", default=None,
                   help="species to make vacancy+Frenkel for; antisite swaps species[0]<->species[1]")
    args = p.parse_args()

    conv = read(str(HOSTS_DIR / f"{args.host}_conv.xyz"))
    super0 = conv.repeat((args.nrep,) * 3)
    species = args.species or sorted(set(super0.get_chemical_symbols()))

    DEFECTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.host}_{args.nrep}x"
    perfect_file = DEFECTS_DIR / f"{tag}_perfect.xyz"
    write(str(perfect_file), super0)

    manifest = []

    def emit(name, atoms, mu_free):
        f = DEFECTS_DIR / f"{tag}_{name}.xyz"
        write(str(f), atoms)
        manifest.append(dict(name=name, file=str(f), perfect=str(perfect_file),
                             mu_free=mu_free, n_atoms=len(atoms)))
        print(f"{name:12s} n={len(atoms):4d} mu_free={mu_free}", flush=True)

    # Frenkel pairs (mu-free): move a central atom into the deepest far void
    for sp in species:
        fr = super0.copy()
        idx = central_atom(fr, sp)
        vacpos = fr.positions[idx].copy()
        del fr[idx]
        void, sep = find_void(fr, vacpos, min_sep=3.0)
        fr += Atoms(sp, positions=[void])
        emit(f"{sp}_frenkel", fr, True)

    # antisite swap (mu-free): species[0] <-> nearest species[1]
    if len(species) >= 2:
        a, b = species[0], species[1]
        asw = super0.copy()
        ia = central_atom(asw, a)
        bidx = [i for i, s in enumerate(asw.get_chemical_symbols()) if s == b]
        jb = bidx[int(np.argmin(np.linalg.norm(
            asw.get_positions()[bidx] - asw.positions[ia], axis=1)))]
        asw.symbols[[ia, jb]] = asw.symbols[[jb, ia]]   # swap species only
        emit(f"antisite_{a}_{b}", asw, True)

    # isolated vacancies (mu-dependent, raw ΔE only)
    for sp in species:
        v = super0.copy()
        del v[central_atom(v, sp)]
        emit(f"{sp}_vacancy", v, False)

    (DEFECTS_DIR / f"{tag}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("→", DEFECTS_DIR / f"{tag}_manifest.json", flush=True)


if __name__ == "__main__":
    main()
