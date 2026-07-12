#!/usr/bin/env python3
"""T0.2 — static point-defect formation energetics on the fluorite host.

Relax each defect cell (fixed cell) and report the energy difference to the perfect
supercell. The mu-free cells (Frenkel pairs, antisite) give clean formation energies;
the isolated vacancies give a raw dE (chemical-potential dependent), reported as-is.

Headline sign test: E_f(O Frenkel) < E_f(U Frenkel) for UO2.

    python t0_defect_static.py --host UO2 --nrep 2 --calc vanilla
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from ase.io import read
from ase.optimize import FIRE

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common_calc import make_calculator  # noqa: E402

DEFECTS_DIR = ROOT / "raddmg" / "defects"
RESULTS_DIR = ROOT / "raddmg" / "results"


def relax_energy(atoms, calc) -> float:
    at = atoms.copy()
    at.calc = calc
    FIRE(at, logfile=None).run(fmax=0.05, steps=300)
    return float(at.get_potential_energy())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--nrep", type=int, default=2)
    p.add_argument("--calc", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.host}_{args.nrep}x"
    manifest = json.loads((DEFECTS_DIR / f"{tag}_manifest.json").read_text())
    calc = make_calculator(args.calc, device=args.device)

    perfect = read(str(DEFECTS_DIR / f"{tag}_perfect.xyz"))
    e_perf = relax_energy(perfect, calc)
    print(f"perfect: E = {e_perf:.4f} eV ({len(perfect)} atoms)", flush=True)

    defects = {}
    for e in manifest:
        cur = read(e["file"])
        e_def = relax_energy(cur, calc)
        dE = e_def - e_perf
        defects[e["name"]] = dict(E=e_def, dE=dE, mu_free=e["mu_free"],
                                  n_atoms=e["n_atoms"])
        kind = "E_f" if e["mu_free"] else "rawΔE(μ-dep)"
        print(f"{e['name']:10s} {kind} = {dE:+.4f} eV", flush=True)

    # headline ordering check (mu-free Frenkel pairs)
    fo = defects.get("O_frenkel", {}).get("dE")
    fu = defects.get("U_frenkel", {}).get("dE")
    if fo is not None and fu is not None:
        ok = fo < fu
        print(f"ordering: O_frenkel ({fo:+.3f}) {'<' if ok else '>='} "
              f"U_frenkel ({fu:+.3f})  -> {'OK (O cheapest)' if ok else 'CHECK'}",
              flush=True)

    out = RESULTS_DIR / f"t0_defect_{args.host}_{args.calc}.json"
    out.write_text(json.dumps(dict(test="t0_defect", host=args.host, calc=args.calc,
                                   nrep=args.nrep, E_perfect=e_perf,
                                   defects=defects), indent=1))
    print("→", out, flush=True)


if __name__ == "__main__":
    main()
