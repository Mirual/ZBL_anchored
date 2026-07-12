#!/usr/bin/env python3
"""Wigner-Seitz defect counting (pure NumPy/scipy; OVITO is not installed).

Given a perfect reference lattice and a current snapshot (same cell + atom count),
assign each current atom to its nearest reference site under PBC. A site with zero
occupants is a vacancy; an extra occupant is an interstitial; a singly-occupied site
whose species differs from the reference site is an antisite. Frenkel pairs = number
of vacancies (== interstitials when the atom count is conserved).

This is the core analysis primitive for the recoil / TDE / cascade tests.

    python wigner_seitz.py --selftest
"""
from __future__ import annotations
import argparse
import itertools
import numpy as np
from scipy.spatial import cKDTree


def _tiled_tree(ref_pos: np.ndarray, cell: np.ndarray):
    """KD-tree over the 3x3x3 periodic images of the reference sites."""
    cell = np.asarray(cell)
    shifts = np.array([i * cell[0] + j * cell[1] + k * cell[2]
                       for i, j, k in itertools.product((-1, 0, 1), repeat=3)])
    nref = len(ref_pos)
    tiled = (ref_pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    return cKDTree(tiled), np.tile(np.arange(nref), len(shifts))


def _nn_distance(ref_pos: np.ndarray, cell: np.ndarray) -> float:
    tree, _ = _tiled_tree(ref_pos, cell)
    d, _ = tree.query(ref_pos, k=2)
    return float(d[:, 1].min())


def analyze(ref_atoms, cur_atoms, cap_frac: float = 0.45) -> dict:
    """ref_atoms = perfect lattice; cur_atoms = snapshot (same cell + atom count).

    Capture-radius Wigner-Seitz: an atom counts as "on" its nearest site only if it is
    within r_cap = cap_frac * (nearest-neighbour distance). Atoms beyond r_cap of every
    site are interstitials; sites with no captured atom are vacancies. The capture radius
    avoids the degeneracy at high-symmetry interstitial holes (e.g. the fluorite
    octahedral site is equidistant to several lattice sites).
    """
    ref_pos = ref_atoms.get_positions()
    ref_sym = np.array(ref_atoms.get_chemical_symbols())
    cur_pos = cur_atoms.get_positions()
    cur_sym = np.array(cur_atoms.get_chemical_symbols())
    cell = np.asarray(ref_atoms.cell[:])

    tree, site_id = _tiled_tree(ref_pos, cell)
    dist, idx = tree.query(cur_pos)
    site = site_id[idx]                       # nearest reference site per current atom
    nref = len(ref_pos)
    nn = _nn_distance(ref_pos, cell)
    r_cap = cap_frac * nn
    captured = dist <= r_cap                  # atom sits on its nearest site

    occ = np.bincount(site[captured], minlength=nref)
    n_vac = int(np.sum(occ == 0))
    n_int = int(np.sum(~captured) + np.sum(np.maximum(occ - 1, 0)))

    antis = 0
    for s in range(nref):
        members = np.where(captured & (site == s))[0]
        if len(members) == 1 and cur_sym[members[0]] != ref_sym[s]:
            antis += 1

    vac_by_species = {}
    for sp in sorted(set(ref_sym.tolist())):
        sp_sites = np.where(ref_sym == sp)[0]
        vac_by_species[sp] = int(np.sum(occ[sp_sites] == 0))

    return dict(
        n_vacancy=n_vac,
        n_interstitial=n_int,
        n_antisite=int(antis),
        n_frenkel=int(min(n_vac, n_int)),
        n_vacancy_by_species=vac_by_species,
        max_disp=float(dist.max()),
        n_displaced=int(np.sum(dist > 0.3 * nn)),
        nn_distance=float(nn),
    )


def _fluorite(cation: str, anion: str, a: float, nrep: int = 2):
    from ase import Atoms
    cat = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
    an = list(itertools.product([.25, .75], repeat=3))
    conv = Atoms([cation] * 4 + [anion] * 8, scaled_positions=cat + an,
                 cell=[a, a, a], pbc=True)
    return conv.repeat((nrep, nrep, nrep))


def _selftest() -> None:
    a = 5.47
    perfect = _fluorite("U", "O", a, nrep=2)
    cur = perfect.copy()
    # move one O from its tetrahedral site to an empty octahedral hole (a/2,a/2,a/2)
    o_idx = [i for i, s in enumerate(cur.get_chemical_symbols()) if s == "O"][0]
    cur.positions[o_idx] = np.array([a / 2, a / 2, a / 2])
    r = analyze(perfect, cur)
    assert r["n_frenkel"] == 1, r
    assert r["n_vacancy_by_species"].get("O", 0) == 1, r
    assert r["n_vacancy_by_species"].get("U", 0) == 0, r
    # identical snapshot → zero defects
    r0 = analyze(perfect, perfect.copy())
    assert r0["n_frenkel"] == 0 and r0["n_vacancy"] == 0, r0
    print("wigner_seitz selftest OK:", r)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
    else:
        p.error("nothing to do; pass --selftest")
