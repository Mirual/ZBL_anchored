#!/usr/bin/env python3
"""Predict vanilla DPA-3.1 + RND-gated pairphys anchor on a single dataset (port of rnd_pairphys_predict.py).

Writes results/vanilla_<cond>.json and results/anchor_<cond>.json (E_ref/E_pred/F_ref/F_pred).
core_zbl=True: below the DimerCache grid (~0.30 Å) — analytic ZBL at full strength (no cap).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpa_common import FOUNDATION, SPLITS, MPTRJ, load_dp, compute_descriptors
from pair_physics import DimerCache, zbl_grad
from gate import RNDGate, smoothstep

RES = Path(__file__).resolve().parents[1] / "results"
DIMER_PKL = RES / "dimer_dpa.pkl"
COND = {"keep_test": (f"{SPLITS}/keep_test.xyz", ":"),
        "u200_test": (f"{SPLITS}/u200_test.xyz", ":"),
        "mptrj": (MPTRJ, "0:400")}


def corr(at, nov, dc, r_lo, r_hi, lam, power, core_zbl=True):
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
        zi, zj = key
        m = ((Zn[i] == zi) & (Zn[j] == zj)) | ((Zn[i] == zj) & (Zn[j] == zi))
        if not m.any() or w[m].max() <= 1e-6:
            continue                       # gate ~0 on all edges of the pair → correction 0, don't build dimer (PROBLEMS.md §6)
        e = dc.get(*key)
        dm = d[m]
        dV = np.interp(dm, e["r"], e["dV"], left=e["dV"][0], right=0.0)
        dVdr = np.interp(dm, e["r"], e["dVdr"], left=e["dVdr"][0], right=0.0)
        sc = lam * w[m]
        if core_zbl:
            core = dm < e["r"][0]
            if core.any():
                vz, dvz = zbl_grad(zi, zj, dm[core])
                dV = dV.copy(); dVdr = dVdr.copy(); sc = sc.copy()
                dV[core] = vz; dVdr[core] = dvz; sc[core] = w[m][core]   # full strength, without lam
        E += 0.5 * (sc * dV).sum()
        np.add.at(F, i[m], (sc * dVdr / dm)[:, None] * D[m])
    return float(E), F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", required=True, choices=list(COND))
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    theta = json.loads((RES / "pairphys_theta.json").read_text())
    r_lo, r_hi, lam, power = theta["r_lo"], theta["r_hi"], theta["lam"], theta["power"]
    calc = load_dp(FOUNDATION)
    gate = RNDGate(dev); dc = DimerCache(calc, cache_path=DIMER_PKL)

    path, sl = COND[args.cond]
    frames = read(path, index=sl, format="extxyz")
    descs = compute_descriptors(frames)
    van, anc, nsk = [], [], 0
    for k, (at, desc) in enumerate(zip(frames, descs)):
        er = float(at.info.get("REF_energy", at.info.get("energy", 0.0)))
        fr = at.arrays.get("REF_forces")
        fr = fr.tolist() if fr is not None else None
        nov = gate.novelty(desc)
        at.calc = calc
        try:
            ev = float(at.get_potential_energy()); fv = np.asarray(at.get_forces())
            ec, fc = corr(at, nov, dc, r_lo, r_hi, lam, power)
            ep, fp = ev + ec, (fv + fc).tolist()
            fvl = fv.tolist()
        except Exception:
            nsk += 1; ev = fvl = ep = fp = None
        van.append({"idx": k, "n_atoms": len(at), "E_ref": er, "E_pred": ev, "F_ref": fr, "F_pred": fvl})
        anc.append({"idx": k, "n_atoms": len(at), "E_ref": er, "E_pred": ep, "F_ref": fr, "F_pred": fp})
    (RES / f"vanilla_{args.cond}.json").write_text(json.dumps({"frames": van}, indent=2))
    (RES / f"anchor_{args.cond}.json").write_text(json.dumps({"frames": anc}, indent=2))
    print(f"{args.cond}: {len(frames)} frames, skipped {nsk} → vanilla_/anchor_{args.cond}.json")


if __name__ == "__main__":
    main()
