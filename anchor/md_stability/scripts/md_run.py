#!/usr/bin/env python3
"""Single MD run: (system, mode, T, ensemble) → trajectory-metrics JSON.
Logs d_min, E, T every log_every steps; detects collapse (d_min<0.5 Å / NaN / exception)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from ase.io import read
from ase import units
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_calculator import AnchorCalculator

COLLAPSE_D = 0.5   # Å — collapse threshold


def dmin(at):
    d = neighbor_list("d", at, 3.0)
    return float(d.min()) if len(d) else 9.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--T", type=float, default=600.0)
    p.add_argument("--ensemble", default="nvt", choices=["nvt", "nve"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--disable-zbl", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    at = read(args.system)
    n = len(at)
    at.calc = AnchorCalculator(mode=args.mode, device="cuda", disable_zbl=args.disable_zbl)
    np.random.seed(args.seed)
    MaxwellBoltzmannDistribution(at, temperature_K=args.T)
    if args.ensemble == "nvt":
        dyn = Langevin(at, args.dt * units.fs, temperature_K=args.T, friction=0.01)
    else:
        dyn = VelocityVerlet(at, args.dt * units.fs)

    log = {"step": [], "E": [], "T": [], "dmin": []}
    E0 = float(at.get_potential_energy())
    collapsed, csteps = False, 0
    nblocks = args.steps // args.log_every
    for blk in range(nblocks):
        try:
            dyn.run(args.log_every)
        except Exception:
            collapsed = True; csteps = blk * args.log_every; break
        step = (blk + 1) * args.log_every
        e = float(at.get_potential_energy()); T = float(at.get_temperature()); dm = dmin(at)
        log["step"].append(step); log["E"].append(e); log["T"].append(T); log["dmin"].append(dm)
        csteps = step
        if (not np.isfinite(e)) or dm < COLLAPSE_D:
            collapsed = True; break

    dm_arr = np.array(log["dmin"]) if log["dmin"] else np.array([dmin(at)])
    T_arr = np.array(log["T"]) if log["T"] else np.array([0.0])
    half = len(T_arr) // 2
    metrics = dict(
        system=Path(args.system).stem, mode=args.mode, T=args.T, ensemble=args.ensemble,
        n_atoms=n, steps_requested=args.steps, steps_run=csteps,
        collapsed=bool(collapsed), survival_steps=csteps if collapsed else args.steps,
        dmin_min=float(dm_arr.min()), dmin_final=float(dm_arr[-1]),
        T_mean=float(T_arr[half:].mean()), T_std=float(T_arr[half:].std()),
        E_drift_per_atom=float((log["E"][-1] - E0) / n) if log["E"] else None,
        log=log,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=1))
    tag = "COLLAPSED@%d" % csteps if collapsed else "survived %d" % args.steps
    print(f"{Path(args.system).stem:14s} {args.mode:10s} {args.ensemble} T={args.T:.0f} "
          f"dmin_min={metrics['dmin_min']:.2f} {tag} → {Path(args.out).name}")


if __name__ == "__main__":
    main()
