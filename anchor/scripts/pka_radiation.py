#!/usr/bin/env python3
"""PKA (primary knock-on atom): give an Sr atom in the structure 2 MeV along its nearest neighbor,
adaptive Velocity-Verlet (small step near close-approach), bounded in time. Comparison
vanilla(ZBL) vs pairphys(ZBL, deploy): do the models physically hold up under a high-energy recoil.

NB: a full cascade (large cell, ns) is unrealistic — this is a bounded simulation of the first high-energy
phase (a few collisions). We track r_min, stability (NaN/blow-up), projectile KE loss, displacements.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from ase import units
from ase.io import read
from ase.neighborlist import neighbor_list
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "md_stability" / "scripts"))
from anchor_calculator import AnchorCalculator   # noqa: E402

CACHE = str(Path(os.environ.get("ZBL_ANCHOR_RESULTS", "results")) / "dimer_tables" / "dimer_zblON_user_wbm.pkl")
KE_EV = 2.0e6
SAFE = 0.02           # step ≤ SAFE·d_min (fraction)
DT_MAX = 0.005 * units.fs
DT_MIN = 1e-7 * units.fs       # small step near the turning point (~0.008 Å at 2 MeV)
T_MAX = 3.0 * units.fs
MAXSTEPS = 80000
CRASH_D = 0.003                # true fusion (below the physical turning-point ~0.008)


def gmin(at):
    d = neighbor_list("d", at, 3.0)
    return float(d.min()) if len(d) else 9.0


def setup_pka(at0):
    at = at0.copy()
    sr = [i for i, s in enumerate(at.get_chemical_symbols()) if s == "Sr"][0]
    # direction — toward the nearest Sr neighbor
    d = at.get_all_distances(mic=True)[sr]; d[sr] = 1e9
    j = int(np.argmin(d))
    vec = at.get_distance(sr, j, mic=True, vector=True); vec /= np.linalg.norm(vec)
    v = np.zeros((len(at), 3)); v[sr] = vec
    at.set_velocities(v)
    ke = at.get_kinetic_energy()
    at.set_velocities(v * np.sqrt(KE_EV / ke))   # rescale exactly to 2 MeV
    return at, sr


def run_pka(calc, at0):
    at, sr = setup_pka(at0); at.calc = calc
    m = at.get_masses()[:, None]
    x0 = at.get_positions().copy()
    v = at.get_velocities(); f = at.get_forces()
    t = 0.0; rmin = gmin(at); pka_ke0 = at.get_kinetic_energy(); ndisp_max = 0
    crash = False
    for step in range(MAXSTEPS):
        a = f / m
        vmax = float(np.linalg.norm(v, axis=1).max())
        dm = gmin(at)
        dt = float(np.clip(SAFE * dm / (vmax + 1e-12), DT_MIN, DT_MAX))
        at.set_positions(at.get_positions() + v * dt + 0.5 * a * dt ** 2)
        try:
            f2 = at.get_forces()
        except Exception:
            crash = True; break
        a2 = f2 / m
        v = v + 0.5 * (a + a2) * dt; at.set_velocities(v); f = f2
        t += dt
        dm = gmin(at); rmin = min(rmin, dm)
        e = at.get_potential_energy()
        if not (np.isfinite(e) and np.isfinite(f).all()) or dm < CRASH_D:
            crash = True; break
        disp = np.linalg.norm(at.get_positions() - x0, axis=1)
        ndisp_max = max(ndisp_max, int((disp > 0.5).sum()))
        if t > T_MAX:
            break
    # projectile KE at the end (projection onto the original axis ≈ total KE of Sr)
    vsr = at.get_velocities()[sr]
    pka_ke_fin = 0.5 * at.get_masses()[sr] * float(vsr @ vsr)
    return dict(steps=step + 1, t_fs=t / units.fs, rmin=rmin, crash=bool(crash),
                pka_ke0_eV=pka_ke0, pka_ke_final_eV=pka_ke_fin, n_displaced=ndisp_max)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", default=os.environ.get("ZBL_PKA_SYSTEMS", "pka_systems"))  # TODO: set for your machine
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cache = CACHE if Path(CACHE).exists() else None
    van = AnchorCalculator(mode="vanilla", device="cuda", disable_zbl=False)
    pp = AnchorCalculator(mode="pairphys", device="cuda", disable_zbl=False, dimer_cache_path=cache)
    import glob
    rows = []
    for fpath in sorted(glob.glob(f"{args.systems}/*.xyz")):
        at0 = read(fpath)
        rv = run_pka(van, at0); rp = run_pka(pp, at0)
        rows.append(dict(system=Path(fpath).stem, n_atoms=len(at0), vanilla=rv, pairphys=rp))
        print(f"{Path(fpath).stem}: vanilla rmin={rv['rmin']:.3f} {'CRASH' if rv['crash'] else 'ok'} "
              f"ndisp={rv['n_displaced']} | pairphys rmin={rp['rmin']:.3f} "
              f"{'CRASH' if rp['crash'] else 'ok'} ndisp={rp['n_displaced']}", flush=True)
    Path(args.out).write_text(json.dumps(dict(KE_eV=KE_EV, rows=rows), indent=1))
    print(f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
