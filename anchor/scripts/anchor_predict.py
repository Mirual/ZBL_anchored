#!/usr/bin/env python3
"""Output-additive physical anchor on top of a pretrained MACE-MH-0 (NO retraining, NO input-gating).

E_total = E_GNN(vanilla) + E_corr,   F_total = F_GNN + F_corr,
where E_corr is an analytic pairwise repulsion, active ONLY at short r and vanishing toward
equilibrium (cutoff). GNN weights are untouched; on normal bonds (>r_b) the correction = 0 →
prediction identical to vanilla (base preserved by construction).

Δ_phys(r) = A·exp(-r/b) · (1 - s(r; r_a, r_b))      # Born–Mayer × (1−smoothstep)
  s — quintic smoothstep: 0 for r<=r_a, 1 for r>=r_b. So V=full for r<=r_a, 0 for r>=r_b.
Forces — analytic gradient (PBC via neighbor_list), continuous (s'(r_a)=s'(r_b)=0).

Parameters A,b (global or per-pair) are calibrated against REAL forces of a distorted set
(scripts/calibrate_anchor.py). This is a generalized-ZBL: ZBL = a special case (short r).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator


def smoothstep(r, a, b):
    t = np.clip((r - a) / (b - a), 0.0, 1.0)
    return ((6.0 * t - 15.0) * t + 10.0) * t * t * t


def dsmoothstep(r, a, b):
    t = np.clip((r - a) / (b - a), 0.0, 1.0)
    return (30.0 * t**4 - 60.0 * t**3 + 30.0 * t**2) / (b - a)


def pair_correction(atoms, A, b, r_a, r_b):
    """E_corr (eV) and F_corr (eV/Å, [N,3]) from Born–Mayer×(1−smoothstep). PBC via neighbor_list."""
    n = len(atoms)
    F = np.zeros((n, 3))
    if A == 0.0:
        return 0.0, F
    i, j, d, D = neighbor_list("ijdD", atoms, r_b)   # each pair twice; D = r_j - r_i (mic)
    if len(d) == 0:
        return 0.0, F
    bm = A * np.exp(-d / b)
    g = 1.0 - smoothstep(d, r_a, r_b)
    V = bm * g
    E = 0.5 * V.sum()
    # dV/dr = bm*(-1/b)*g + bm*(-dsmoothstep)
    dVdr = bm * (-1.0 / b) * g + bm * (-dsmoothstep(d, r_a, r_b))
    # F_i = (dV/dr) * D/|D|   (repulsion: dVdr<0 → force away from j)
    fij = (dVdr / d)[:, None] * D                    # contribution to force on i from pair (i,j)
    np.add.at(F, i, fij)
    return float(E), F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model"))
    p.add_argument("--head", default="mp_pbe_refit_add")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--A", type=float, default=0.0, help="Born–Mayer amplitude (eV); 0 = vanilla only")
    p.add_argument("--b", type=float, default=0.3, help="Born–Mayer scale (Å)")
    p.add_argument("--ra", type=float, default=0.3)
    p.add_argument("--rb", type=float, default=1.5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    calc = MACECalculator(model_paths=[args.model], device=args.device,
                          default_dtype="float32", head=args.head)
    frames = read(args.data, index=":", format="extxyz")
    recs, nsk = [], 0
    for k, at in enumerate(frames):
        er = float(at.info.get("REF_energy", at.info.get("energy", 0.0)))
        fr = at.arrays.get("REF_forces")
        at.calc = calc
        try:
            ev = float(at.get_potential_energy()); fv = np.asarray(at.get_forces())
            ec, fc = pair_correction(at, args.A, args.b, args.ra, args.rb)
            ep = ev + ec; fp = (fv + fc).tolist()
        except Exception:
            nsk += 1; ep, fp = None, None
        recs.append({"idx": k, "n_atoms": len(at), "E_ref": er, "E_pred": ep,
                     "F_ref": fr.tolist() if fr is not None else None, "F_pred": fp})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"frames": recs,
        "anchor": {"A": args.A, "b": args.b, "ra": args.ra, "rb": args.rb}}, indent=2))
    print(f"{len(frames)} frames, skipped {nsk}, anchor A={args.A} b={args.b} ra={args.ra} rb={args.rb} → {args.out}")


if __name__ == "__main__":
    main()
