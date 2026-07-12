#!/usr/bin/env python3
"""Replication of MLIP-Arena "high-pressure stability" (simplified): isotropic cell compression V/V0
from 1.0 downward in steps; at each step a short FIRE relaxation of positions (cell fixed); we record
energy/atom, d_min, force finiteness. The "crash point" = the first compression where d_min<0.5 Å / NaN /
energy blow-up (>wall). Comparison of vanilla / pairphys / vanilla-noZBL — shows the role of ZBL/anchor.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "md_stability" / "scripts"))
from anchor_calculator import AnchorCalculator  # noqa: E402

CACHE = str(Path(os.environ.get("ZBL_ANCHOR_RESULTS", "results")) / "dimer_tables" / "snapshot_diatomics.pkl")


def dmin(at):
    d = neighbor_list("d", at, 3.0); return float(d.min()) if len(d) else 9.0


def ramp(calc, at0, scales, steps=40):
    out = []
    for s in scales:
        at = at0.copy()
        at.set_cell(at0.cell * s, scale_atoms=True)
        at.calc = calc
        try:
            FIRE(at, logfile=None).run(fmax=0.1, steps=steps)
            e = float(at.get_potential_energy()) / len(at)
            f = np.asarray(at.get_forces()); fok = np.isfinite(f).all()
            dm = dmin(at)
            crash = (not np.isfinite(e)) or (not fok) or dm < 0.5 or abs(e) > 1e4
        except Exception:
            e, dm, crash = None, None, True
        out.append(dict(scale=float(s), e_per_atom=e, dmin=dm, crash=bool(crash)))
        if out[-1]["crash"]:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="vanilla", choices=["vanilla", "bornmayer", "pairphys"])
    p.add_argument("--disable-zbl", action="store_true")
    p.add_argument("--system", required=True); p.add_argument("--out", required=True)
    args = p.parse_args()
    calc = AnchorCalculator(mode=args.mode, device="cuda", disable_zbl=args.disable_zbl,
                            dimer_cache_path=CACHE if args.mode == "pairphys" else None)
    at0 = read(args.system)
    scales = np.linspace(1.0, 0.55, 19)   # V/V0 from 1 to 0.55^3≈0.17 (extreme compression)
    res = ramp(calc, at0, scales)
    last = res[-1]
    Path(args.out).write_text(json.dumps(dict(mode=args.mode, disable_zbl=args.disable_zbl,
                                              system=Path(args.system).stem, ramp=res), indent=1))
    tag = "CRASH@scale=%.3f (dmin=%.2f)" % (last["scale"], last["dmin"] or 0) if last["crash"] else "survived to %.3f" % scales[-1]
    print(f"{args.mode}{'(noZBL)' if args.disable_zbl else ''}: {tag} → {args.out}")


if __name__ == "__main__":
    main()
