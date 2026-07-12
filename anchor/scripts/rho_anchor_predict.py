#!/usr/bin/env python3
"""ρ-gated physical anchor: the correction turns on by RELIABILITY (extrapolation-score), not by r.

ρ_i = smoothstep(kNN-dist of atom i's descriptor to the MPtrj reference; r_lo, r_hi) ∈ [0,1].
w_pair = max(ρ_i, ρ_j) — a pair is corrected if at least one atom is OOD.
E = E_vanilla + Σ w_pair·V_BM(r);  F = F_vanilla + Σ w_pair·F_BM   (w as a fixed scalar at inference).

Normal bond (in-dist) → ρ≈0 → no correction → MPtrj = vanilla.
Compressed contact (OOD) → ρ≈1 → correction → keep gets fixed.

Modes:  --check (distribution of ρ keep vs MPtrj) | predict (--data --out).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_predict import smoothstep  # noqa: E402

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
REF = Path(__file__).resolve().parents[1] / "results" / "rho_reference.npz"


class Rho:
    def __init__(self):
        z = np.load(REF)
        self.mean, self.comp, ref = z["pca_mean"], z["pca_comp"], z["ref"]
        self.r_lo, self.r_hi, self.k = float(z["r_lo"]), float(z["r_hi"]), int(z["knn"])
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(ref)

    def raw_dist(self, desc):                 # desc [N,256] → kNN mean dist [N] (before smoothstep)
        proj = (desc - self.mean) @ self.comp.T
        dist, _ = self.nn.kneighbors(proj)
        return dist.mean(1)

    def of(self, desc):                       # desc [N,256] → ρ [N]
        return smoothstep(self.raw_dist(desc), self.r_lo, self.r_hi)


def pair_corr_gated(atoms, rho, A, b, r_a, r_b, power=1.0):
    n = len(atoms); F = np.zeros((n, 3))
    i, j, d, D = neighbor_list("ijdD", atoms, r_b)
    if len(d) == 0:
        return 0.0, F
    from anchor_predict import dsmoothstep
    w = np.maximum(rho[i], rho[j]) ** power   # per-pair gate, w=ρ^power (steeper → fewer borderline)
    bm = A * np.exp(-d / b); g = 1.0 - smoothstep(d, r_a, r_b)
    V = w * bm * g
    E = 0.5 * V.sum()
    dVdr = w * (bm * (-1.0 / b) * g + bm * (-dsmoothstep(d, r_a, r_b)))
    np.add.at(F, i, (dVdr / d)[:, None] * D)
    return float(E), F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--data"); p.add_argument("--out")
    p.add_argument("--A", type=float, default=800.0); p.add_argument("--b", type=float, default=0.5)
    p.add_argument("--ra", type=float, default=0.3); p.add_argument("--rb", type=float, default=1.5)
    p.add_argument("--power", type=float, default=1.0, help="gate w=ρ^power (higher → steeper)")
    p.add_argument("--rho-lo", type=float, default=None, help="override ρ-threshold r_lo")
    p.add_argument("--rho-hi", type=float, default=None, help="override ρ-threshold r_hi")
    args = p.parse_args()
    calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
    rho = Rho()
    if args.rho_lo is not None:
        rho.r_lo = args.rho_lo
    if args.rho_hi is not None:
        rho.r_hi = args.rho_hi

    if args.check:
        PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
        keep = read(PRE / "splits" / "keep_test.xyz", index=":")
        mp = read(os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), index="0:300")
        rk = np.concatenate([rho.of(np.asarray(calc.get_descriptors(a))) for a in keep])
        rm = np.concatenate([rho.of(np.asarray(calc.get_descriptors(a))) for a in mp])
        print(f"ρ on keep:  med={np.median(rk):.3f}  mean={rk.mean():.3f}  frac(ρ>0.5)={(rk>0.5).mean()*100:.0f}%")
        print(f"ρ on MPtrj: med={np.median(rm):.3f}  mean={rm.mean():.3f}  frac(ρ>0.5)={(rm>0.5).mean()*100:.0f}%")
        np.savez(Path(__file__).resolve().parents[1] / "results" / "rho_check.npz", rk=rk, rm=rm)
        return

    frames = read(args.data, index=":", format="extxyz")
    recs, nsk = [], 0
    for k, at in enumerate(frames):
        er = float(at.info.get("REF_energy", at.info.get("energy", 0.0)))
        fr = at.arrays.get("REF_forces")
        r = None if args.A == 0 else rho.of(np.asarray(calc.get_descriptors(at)))  # A=0 → pure vanilla
        at.calc = calc
        try:
            ev = float(at.get_potential_energy()); fv = np.asarray(at.get_forces())
            if args.A == 0:
                ec, fc = 0.0, 0.0
            else:
                ec, fc = pair_corr_gated(at, r, args.A, args.b, args.ra, args.rb, args.power)
            ep = ev + ec; fp = (fv + fc).tolist()
        except Exception:
            nsk += 1; ep, fp = None, None
        recs.append({"idx": k, "n_atoms": len(at), "E_ref": er, "E_pred": ep,
                     "F_ref": fr.tolist() if fr is not None else None, "F_pred": fp})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"frames": recs}, indent=2))
    print(f"{len(frames)} frames skipped {nsk} → {args.out}")


if __name__ == "__main__":
    main()
