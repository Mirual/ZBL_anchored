#!/usr/bin/env python3
"""Predict: RND-gate + per-pair physics (DimerCache residual) × λ. For held-out comparison with global Born-Mayer."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
from mace.calculators import MACECalculator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_predict import smoothstep
from pair_physics import DimerCache
from rnd_anchor_predict import RNDGate

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")


def corr(at, nov, dc, r_lo, r_hi, lam, power, core_zbl=False):
    """core_zbl=True: below r_floor (DimerCache grid, ~0.30 Å) use the analytic ZBL
    (diverges as 1/r, exact at any r) at FULL strength (gated by w, without lam) — many-body
    screening at the core is negligible. This way the anchor also covers the keV–MeV regime without a ceiling."""
    from pair_physics import zbl_grad
    F = np.zeros((len(at), 3)); Zn = at.numbers
    Zs = np.unique(Zn)
    rmax = max(dc.kappa * (covalent_radii[a] + covalent_radii[b]) + dc.width for a in Zs for b in Zs)
    i, j, d, D = neighbor_list("ijdD", at, float(rmax))
    if len(d) == 0:
        return 0.0, F
    rho = smoothstep(nov, r_lo, r_hi)
    w = np.maximum(rho[i], rho[j]) ** power
    E = 0.0
    for key in {(min(Zn[a], Zn[b]), max(Zn[a], Zn[b])) for a, b in zip(i, j)}:
        e = dc.get(*key); zi, zj = key
        m = ((Zn[i] == zi) & (Zn[j] == zj)) | ((Zn[i] == zj) & (Zn[j] == zi))
        dm = d[m]
        dV = np.interp(dm, e["r"], e["dV"], left=e["dV"][0], right=0.0)
        dVdr = np.interp(dm, e["r"], e["dVdr"], left=e["dVdr"][0], right=0.0)
        sc = lam * w[m]
        if core_zbl:
            core = dm < e["r"][0]                    # r_floor = first grid point (~0.30 Å)
            if core.any():
                vz, dvz = zbl_grad(zi, zj, dm[core])
                dV = dV.copy(); dVdr = dVdr.copy(); sc = sc.copy()
                dV[core] = vz; dVdr[core] = dvz; sc[core] = w[m][core]   # full strength, without lam
        E += 0.5 * (sc * dV).sum()
        np.add.at(F, i[m], (sc * dVdr / dm)[:, None] * D[m])
    return float(E), F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--rho-lo", type=float, default=0.05); p.add_argument("--rho-hi", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=0.4); p.add_argument("--power", type=float, default=2.0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(model_paths=[VAN], device=dev, default_dtype="float32", head="mp_pbe_refit_add")
    gate = RNDGate(dev); dc = DimerCache(calc)
    recs, nsk = [], 0
    for k, at in enumerate(read(args.data, index=":", format="extxyz")):
        er = float(at.info.get("REF_energy", at.info.get("energy", 0.0)))
        fr = at.arrays.get("REF_forces")
        nov = gate.novelty(calc.get_descriptors(at))
        at.calc = calc
        try:
            ev = float(at.get_potential_energy()); fv = np.asarray(at.get_forces())
            ec, fc = corr(at, nov, dc, args.rho_lo, args.rho_hi, args.lam, args.power)
            ep = ev + ec; fp = (fv + fc).tolist()
        except Exception:
            nsk += 1; ep, fp = None, None
        recs.append({"idx": k, "n_atoms": len(at), "E_ref": er, "E_pred": ep,
                     "F_ref": fr.tolist() if fr is not None else None, "F_pred": fp})
    Path(args.out).write_text(json.dumps({"frames": recs}, indent=2))
    print(f"skipped {nsk} → {args.out}")


if __name__ == "__main__":
    main()
