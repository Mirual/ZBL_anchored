#!/usr/bin/env python3
"""Radiation regime (up to 2 MeV): comparison of vanilla(ZBL) vs pairphys(ZBL, deploy).

2 MeV dynamics cannot be resolved with an fs step (v~300 Å/fs) → static scan of V(r) + turning point
r_min(E)=min r: V(r)≥E (KE→PE in a head-on collision). Hypothesis: at 2 MeV both models rely on the
BUILT-IN ZBL (for pairphys the residual≈0 at the core, since the model already has ZBL → no double-count) →
the turning-point is identical. I.e. in the radiation regime the anchor neither hurts nor helps — ZBL does its job.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as AN
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "md_stability" / "scripts"))
from anchor_calculator import AnchorCalculator        # noqa: E402
from dimer_scan import find_rmin                        # noqa: E402

ENERGIES = [200.0, 2e3, 2e4, 2e5, 2e6]
CACHE = str(Path(os.environ.get("ZBL_ANCHOR_RESULTS", "results")) / "dimer_tables" / "dimer_zblON_user_wbm.pkl")


def Vcurve(calc, Zi, Zj, rs, rref=6.0):
    def E(r):
        at = Atoms(numbers=[Zi, Zj], positions=[[0, 0, 0], [r, 0, 0]], cell=[30] * 3, pbc=False)
        at.calc = calc; return float(at.get_potential_energy())
    eref = E(rref)
    out = []
    for r in rs:
        try:
            out.append(E(r) - eref)
        except Exception:
            out.append(None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="Sr-Sr,Ti-O,O-O")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cache = CACHE if Path(CACHE).exists() else None
    calcs = {"vanilla_ZBL": AnchorCalculator(mode="vanilla", device="cuda", disable_zbl=False),
             "pairphys_ZBL": AnchorCalculator(mode="pairphys", device="cuda", disable_zbl=False,
                                              dimer_cache_path=cache)}
    rs = np.logspace(np.log10(0.002), np.log10(2.0), 80).tolist()
    res = {}
    for pr in args.pairs.split(","):
        a, b = pr.split("-"); Zi, Zj = AN[a], AN[b]
        res[pr] = {}
        for tag, c in calcs.items():
            V = Vcurve(c, Zi, Zj, rs)
            rmin = {f"{e:.0f}": find_rmin(rs, V, e) for e in ENERGIES}
            res[pr][tag] = dict(rmin=rmin)
        print(f"{pr}: 2MeV turning -> vanilla {res[pr]['vanilla_ZBL']['rmin']['2000000']}, "
              f"pairphys {res[pr]['pairphys_ZBL']['rmin']['2000000']}", flush=True)
    Path(args.out).write_text(json.dumps(dict(r=rs, energies=ENERGIES, pairs=res), indent=1))
    print(f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
