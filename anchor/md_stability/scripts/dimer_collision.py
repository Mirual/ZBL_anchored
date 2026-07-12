#!/usr/bin/env python3
"""Controlled head-on dimer collision: two atoms fly toward each other with fixed KE.
Minimum approach r_min = a clean test of short-range repulsion (no thermostat, NVE).
softer repulsion → atoms get closer/pass through; anchor/ZBL → earlier bounce.

Comparison: vanilla(ZBL on) | vanilla(ZBL off) | bornmayer(ZBL off) | pairphys(ZBL off).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_calculator import AnchorCalculator


def collide(Zi, Zj, mode, disable_zbl, ke_ev, d0=3.0, steps=600, dt=0.5):
    at = Atoms(numbers=[Zi, Zj], positions=[[-d0 / 2, 0, 0], [d0 / 2, 0, 0]], cell=[30] * 3, pbc=False)
    m = at.get_masses()
    # head-on: equal momentum toward each other, total relative KE = ke_ev
    # KE_rel = 0.5*mu*v_rel^2 ; v_rel = v0*2 (each v0 toward center)
    mu = m[0] * m[1] / (m[0] + m[1])
    v_rel = np.sqrt(2 * ke_ev / mu) * np.sqrt(units.eV / (units._amu / units.kg) / (units.m / units.second) ** 2) ** 0  # use ASE units below
    # simpler: in ASE eV, masses in amu, v in Å/(ASE time). KE=0.5 m v^2 [eV] with m in amu·(units), v in Å/fs*units.fs
    # set v_rel from KE_rel = 0.5*mu*v_rel^2 in ASE units:
    v_rel = np.sqrt(2 * ke_ev / mu)            # Å/(ASE time unit)
    v0 = v_rel * m[1] / (m[0] + m[1]); v1 = v_rel * m[0] / (m[0] + m[1])
    vel = np.zeros((2, 3)); vel[0, 0] = +v0; vel[1, 0] = -v1
    at.set_velocities(vel)
    at.calc = AnchorCalculator(mode=mode, device="cuda", disable_zbl=disable_zbl)
    dyn = VelocityVerlet(at, dt * units.fs)
    rmin = d0
    for _ in range(steps):
        try:
            dyn.run(1)
        except Exception:
            break
        r = at.get_distance(0, 1)
        rmin = min(rmin, r)
        if not np.isfinite(r) or r < 0.15:
            rmin = float(r) if np.isfinite(r) else 0.0; break
    return float(rmin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="O-O")
    p.add_argument("--ke", type=float, default=60.0, help="relative KE, eV")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    from ase.data import atomic_numbers as AN
    a, b = args.pair.split("-"); Zi, Zj = AN[a], AN[b]
    print(f"=== {args.pair} head-on collision, KE_rel={args.ke} eV ===")
    cfg = [("vanilla ZBL-on", "vanilla", False),
           ("vanilla ZBL-off", "vanilla", True),
           ("bornmayer ZBL-off", "bornmayer", True),
           ("pairphys ZBL-off", "pairphys", True)]
    res = {}
    for lab, mode, dz in cfg:
        rmin = collide(Zi, Zj, mode, dz, args.ke)
        res[lab] = rmin
        print(f"  {lab:20s}: r_min = {rmin:.3f} Å  {'(THROUGH/collapse)' if rmin < 0.4 else ''}")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(pair=args.pair, ke=args.ke, rmin=res), indent=1))


if __name__ == "__main__":
    main()
