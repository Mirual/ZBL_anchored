#!/usr/bin/env python3
"""T0.1 — equation of state + bulk modulus on the perfect fluorite host.

Scan V/V0 over an isotropic strain range, relax atomic positions at each fixed volume,
fit Birch-Murnaghan -> B0. Sanity that the model gives a mechanically sound crystal and
that the anchor does not shift the equilibrium EOS vs vanilla.

    python t0_eos_elastic.py --hosts UO2 ThO2 NpO2 --calc vanilla
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.optimize import FIRE

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common_calc import make_calculator  # noqa: E402

HOSTS_DIR = ROOT / "raddmg" / "hosts"
RESULTS_DIR = ROOT / "raddmg" / "results"


def eos_one(conv, calc, npoints: int, vmin: float, vmax: float) -> dict:
    cell0 = conv.cell.copy()
    vols, ens = [], []
    for v in np.linspace(vmin, vmax, npoints):
        at = conv.copy()
        at.calc = calc
        at.set_cell(cell0 * (v ** (1.0 / 3.0)), scale_atoms=True)
        FIRE(at, logfile=None).run(fmax=0.05, steps=200)
        vols.append(float(at.get_volume()))
        ens.append(float(at.get_potential_energy()))
    out = dict(volumes=vols, energies=ens, n_atoms=len(conv))
    try:
        from ase.eos import EquationOfState
        from ase.units import GPa
        eos = EquationOfState(vols, ens)
        v0, e0, B = eos.fit()
        out.update(v0=float(v0), e0=float(e0), B0_GPa=float(B / GPa))
    except Exception as e:  # fit can fail on a noisy/monotone curve
        out.update(v0=None, e0=None, B0_GPa=None, fit_error=str(e))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hosts", nargs="+", default=["UO2", "ThO2", "NpO2"])
    p.add_argument("--calc", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--npoints", type=int, default=11)
    p.add_argument("--vmin", type=float, default=0.92)
    p.add_argument("--vmax", type=float, default=1.08)
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    calc = make_calculator(args.calc, device=args.device)
    res = {}
    for host in args.hosts:
        conv = read(str(HOSTS_DIR / f"{host}_conv.xyz"))
        r = eos_one(conv, calc, args.npoints, args.vmin, args.vmax)
        res[host] = r
        b = r.get("B0_GPa")
        print(f"{host}: B0 = {b:.1f} GPa" if b else f"{host}: B0 fit failed", flush=True)
    out = RESULTS_DIR / f"t0_eos_{args.calc}.json"
    hosts = {}
    if out.exists():
        try:
            hosts = json.loads(out.read_text()).get("hosts", {})
        except Exception:
            hosts = {}
    hosts.update(res)  # merge new hosts, keep previously computed ones
    out.write_text(json.dumps(dict(test="t0_eos", calc=args.calc, hosts=hosts), indent=1))
    print("→", out, flush=True)


if __name__ == "__main__":
    main()
