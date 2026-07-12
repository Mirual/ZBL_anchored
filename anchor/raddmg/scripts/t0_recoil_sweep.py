#!/usr/bin/env python3
"""T0.3 — low-energy recoil sweep (defect-regime, the gap the existing tests skip).

Generalizes pka_radiation.run_pka into a (species x direction x energy) sweep. For each
case: give one atom a recoil velocity along a crystallographic direction, run bounded
adaptive-dt NVE, quench (FIRE) to 0 K, count surviving Frenkel pairs with Wigner-Seitz.
Writes one row per case incrementally so partial sweeps survive.

    python t0_recoil_sweep.py --host UO2 --calc vanilla --species U O \
        --dirs 100 110 111 --energies 10 20 40 75 150 300 600 1000 --nrep 3
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common_calc import make_calculator   # noqa: E402
from wigner_seitz import analyze          # noqa: E402

HOSTS_DIR = ROOT / "raddmg" / "hosts"
RESULTS_DIR = ROOT / "raddmg" / "results"

DIRS = {"100": (1, 0, 0), "110": (1, 1, 0), "111": (1, 1, 1)}
SAFE = 0.05
DT_MAX = 1.0 * units.fs
DT_MIN = 1e-6 * units.fs
CRASH_D = 0.02


def gmin(at) -> float:
    d = neighbor_list("d", at, 3.0)
    return float(d.min()) if len(d) else 9.0


def run_recoil(calc, atoms0, pka_idx, direction, energy_eV, t_max_fs, maxsteps):
    at = atoms0.copy()
    at.calc = calc
    u = np.asarray(direction, float)
    u /= np.linalg.norm(u)
    v = np.zeros((len(at), 3))
    v[pka_idx] = u
    at.set_velocities(v)
    ke = at.get_kinetic_energy()
    at.set_velocities(v * np.sqrt(energy_eV / ke))   # scale to exact recoil energy

    m = at.get_masses()[:, None]
    x0 = at.get_positions().copy()
    v = at.get_velocities()
    f = at.get_forces()
    t = 0.0
    rmin = gmin(at)
    ndisp = 0
    crash = False
    t_max = t_max_fs * units.fs
    step = 0
    for step in range(maxsteps):
        a = f / m
        vmax = float(np.linalg.norm(v, axis=1).max())
        dm = gmin(at)
        dt = float(np.clip(SAFE * dm / (vmax + 1e-12), DT_MIN, DT_MAX))
        at.set_positions(at.get_positions() + v * dt + 0.5 * a * dt ** 2)
        try:
            f2 = at.get_forces()
        except Exception:
            crash = True
            break
        a2 = f2 / m
        v = v + 0.5 * (a + a2) * dt
        at.set_velocities(v)
        f = f2
        t += dt
        dm = gmin(at)
        rmin = min(rmin, dm)
        e = at.get_potential_energy()
        if not (np.isfinite(e) and np.isfinite(f).all()) or dm < CRASH_D:
            crash = True
            break
        ndisp = max(ndisp, int((np.linalg.norm(at.get_positions() - x0, axis=1) > 0.5).sum()))
        if t > t_max:
            break
    return at, dict(rmin=float(rmin), crash=bool(crash), steps=step + 1,
                    t_fs=t / units.fs, n_displaced_dyn=ndisp)


def quench(calc, atoms):
    at = atoms.copy()
    at.calc = calc
    at.set_velocities(np.zeros((len(at), 3)))
    FIRE(at, logfile=None).run(fmax=0.1, steps=300)
    return at


def central_atom(atoms, species: str) -> int:
    center = atoms.cell[:].sum(axis=0) / 2.0
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == species]
    d = np.linalg.norm(atoms.get_positions()[idx] - center, axis=1)
    return idx[int(np.argmin(d))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--calc", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--species", nargs="+", default=None,
                   help="default: cation + O")
    p.add_argument("--dirs", nargs="+", default=["100", "110", "111"])
    p.add_argument("--energies", nargs="+", type=float,
                   default=[10, 20, 40, 75, 150, 300, 600, 1000])
    p.add_argument("--nrep", type=int, default=3)
    p.add_argument("--t-max-fs", type=float, default=1000.0)
    p.add_argument("--maxsteps", type=int, default=40000)
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conv = read(str(HOSTS_DIR / f"{args.host}_conv.xyz"))
    perfect = conv.repeat((args.nrep, args.nrep, args.nrep))
    syms = perfect.get_chemical_symbols()
    cation = next(s for s in syms if s != "O")
    species = args.species or [cation, "O"]

    calc = make_calculator(args.calc, device=args.device)
    out = RESULTS_DIR / f"t0_recoil_{args.host}_{args.calc}.json"
    rows = []
    print(f"{args.host} {args.calc}: {len(perfect)} atoms, "
          f"species={species} dirs={args.dirs} E={args.energies}", flush=True)

    for sp in species:
        pka = central_atom(perfect, sp)
        for dname in args.dirs:
            for E in args.energies:
                try:
                    final, dyn = run_recoil(calc, perfect, pka, DIRS[dname], E,
                                            args.t_max_fs, args.maxsteps)
                    relaxed = quench(calc, final)
                    ws = analyze(perfect, relaxed)
                    row = dict(species=sp, direction=dname, energy_eV=E,
                               **dyn,
                               n_frenkel=ws["n_frenkel"],
                               n_vacancy_by_species=ws["n_vacancy_by_species"],
                               n_interstitial=ws["n_interstitial"],
                               max_disp=ws["max_disp"])
                except Exception as ex:
                    row = dict(species=sp, direction=dname, energy_eV=E,
                               error=str(ex))
                rows.append(row)
                fp = row.get("n_frenkel", "ERR")
                print(f"  {sp} <{dname}> {E:6.0f} eV -> rmin={row.get('rmin', float('nan')):.3f} "
                      f"FP={fp} {'CRASH' if row.get('crash') else ''}", flush=True)
                out.write_text(json.dumps(dict(test="t0_recoil", host=args.host,
                                               calc=args.calc, nrep=args.nrep,
                                               n_atoms=len(perfect), rows=rows), indent=1))
    print("→", out, flush=True)


if __name__ == "__main__":
    main()
