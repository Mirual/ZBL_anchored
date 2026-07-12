#!/usr/bin/env python3
"""Replication of the MLIP-Arena "homonuclear diatomics" protocol on our AnchorCalculator.

For each element: homonuclear dimer, r from 0.9·r_cov to 3.1·r_vdW (as in MLIP-Arena),
compute E(r), F(r) for vanilla and pairphys (with a cache table). Three physics metrics (as in Arena):
  1. conservativeness — how well F = −dE/dr (max deviation of the analytic force from the numerical derivative);
  2. short-range stiffness — is the wall repulsive (E grows toward small r; no "holes"/spurious wells below the minimum);
  3. smoothness — number of spurious extrema of dE/dr (besides the one true minimum).
Comparison of vanilla vs pairphys. pairphys uses the saved table (--cache).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, vdw_radii, atomic_numbers, chemical_symbols

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "md_stability" / "scripts"))
from anchor_calculator import AnchorCalculator  # noqa: E402


def pec(calc, Z, n=60):
    rc = covalent_radii[Z]; rv = vdw_radii[Z]
    rv = rv if np.isfinite(rv) else 2.0 * rc
    rs = np.linspace(0.9 * rc, 3.1 * rv, n)
    E = np.full(n, np.nan); Fr = np.full(n, np.nan)  # F along the bond (on atom 1, projection onto the axis)
    for k, r in enumerate(rs):
        at = Atoms(numbers=[Z, Z], positions=[[0, 0, 0], [r, 0, 0]], cell=[30, 30, 30], pbc=False)
        at.calc = calc
        try:
            E[k] = at.get_potential_energy()
            Fr[k] = float(at.get_forces()[1, 0])   # force on atom 2 along x (>0 = repulsion)
        except Exception:
            pass
    return rs, E, Fr


def metrics(rs, E, Fr):
    m = np.isfinite(E) & np.isfinite(Fr)
    rs, E, Fr = rs[m], E[m], Fr[m]
    if len(rs) < 5:
        return None
    # 1. conservativeness: F_analytic vs −dE/dr (numerically). F on atom2 along +x = −dE/dr.
    dEdr = np.gradient(E, rs)
    cons = float(np.nanmax(np.abs(Fr - (-dEdr))) / (np.nanmax(np.abs(Fr)) + 1e-9))
    # 2. short-range stiffness: E at minimal r minus min E (wall height, eV) + "holes"
    imin = int(np.argmin(E))
    wall = float(E[0] - E[imin])                       # >0 = a repulsive wall exists
    holes = int(np.sum(np.diff(E[:imin + 1]) > 0))     # below the minimum E must GROW toward small r (decreasing index) → violations = holes
    # 3. smoothness: spurious extrema of dE/dr (besides the sign change at the true minimum)
    sign_changes = int(np.sum(np.diff(np.sign(dEdr)) != 0))
    return dict(conservativeness=cons, wall_eV=wall, holes=holes, extrema=sign_changes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="vanilla", choices=["vanilla", "pairphys"])
    p.add_argument("--cache", default=None, help="dimer table for pairphys")
    p.add_argument("--out", required=True)
    p.add_argument("--elements", default=None, help="space-separated; default = from cache / 19 user")
    args = p.parse_args()
    if args.elements:
        Zs = [atomic_numbers[s] for s in args.elements.split()]
    elif args.cache and Path(args.cache).exists():
        import pickle
        c = pickle.loads(Path(args.cache).read_bytes())
        Zs = sorted({zi for zi, zj in c if zi == zj})
    else:
        Zs = [atomic_numbers[s] for s in "Cl Cs F Fe Li Mg Mn N O P S Sr Ti Y Zr".split()]
    calc = AnchorCalculator(mode=args.mode, device="cuda", disable_zbl=False,
                            dimer_cache_path=args.cache if args.mode == "pairphys" else None)
    res = {}; t0 = time.time()
    for Z in Zs:
        rs, E, Fr = pec(calc, Z)
        mt = metrics(rs, E, Fr)
        if mt:
            res[chemical_symbols[Z]] = mt
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"{args.mode}: {len(res)} elements, {time.time()-t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
