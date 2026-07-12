#!/usr/bin/env python3
"""RND-gated physical anchor: correction gate by RND-novelty (instead of kNN-ρ).
--check : distribution of novelty keep vs u200 vs MPtrj (does it separate?).
predict : vanilla + Σ w(novelty)·V_BM(r), w=smoothstep(novelty; r_lo,r_hi)^power.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from ase.io import read
from mace.calculators import MACECalculator
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_predict import smoothstep
from rho_anchor_predict import pair_corr_gated

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
RND = Path(__file__).resolve().parents[1] / "results" / "rnd.pt"


def mlp(din, demb):
    return nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, demb))


class RNDGate:
    def __init__(self, dev):
        z = torch.load(RND, map_location=dev, weights_only=False)
        self.mu = z["mu"]; self.sd = z["sd"]; self.dev = dev
        self.r_lo, self.r_hi = z["r_lo"], z["r_hi"]
        self.target = mlp(z["din"], z["demb"]).to(dev); self.target.load_state_dict(z["target"]); self.target.eval()
        self.pred = mlp(z["din"], z["demb"]).to(dev); self.pred.load_state_dict(z["predictor"]); self.pred.eval()

    def novelty(self, desc):                       # desc [N,256] → per-atom novelty [N]
        x = torch.tensor((np.asarray(desc) - self.mu) / self.sd, device=self.dev, dtype=torch.float32)
        with torch.no_grad():
            return ((self.pred(x) - self.target(x)) ** 2).mean(1).cpu().numpy()

    def of(self, desc, r_lo=None, r_hi=None):      # → ρ∈[0,1]
        return smoothstep(self.novelty(desc), r_lo or self.r_lo, r_hi or self.r_hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--data"); p.add_argument("--out")
    p.add_argument("--A", type=float, default=800.0); p.add_argument("--b", type=float, default=0.5)
    p.add_argument("--ra", type=float, default=0.3); p.add_argument("--rb", type=float, default=1.5)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--rho-lo", type=float, default=None); p.add_argument("--rho-hi", type=float, default=None)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(model_paths=[VAN], device=dev, default_dtype="float32", head="mp_pbe_refit_add")
    g = RNDGate(dev)
    rlo = args.rho_lo or g.r_lo; rhi = args.rho_hi or g.r_hi

    if args.check:
        PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
        srcs = {"keep": (PRE/"splits"/"keep_test.xyz", ":"),
                "u200": (PRE/"splits"/"u200_test.xyz", ":"),
                "mptrj": (os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), "0:300")}
        print(f"calibration: r_lo(p90 MPtrj)={g.r_lo:.4f}  r_hi(p99.5)={g.r_hi:.4f}\n")
        for name, (path, sl) in srcs.items():
            nov = np.concatenate([g.novelty(calc.get_descriptors(a)) for a in read(path, index=sl)])
            rho = smoothstep(nov, rlo, rhi)
            print(f"{name:6s}: novelty med={np.median(nov):.4f} p90={np.percentile(nov,90):.4f} "
                  f"max={nov.max():.4f} | ρ>0.5 on {(rho>0.5).mean()*100:4.0f}% of atoms  mean ρ={rho.mean():.3f}")
        return

    frames = read(args.data, index=":", format="extxyz")
    recs, nsk = [], 0
    for k, at in enumerate(frames):
        er = float(at.info.get("REF_energy", at.info.get("energy", 0.0)))
        fr = at.arrays.get("REF_forces")
        rho = g.of(calc.get_descriptors(at), rlo, rhi)
        at.calc = calc
        try:
            ev = float(at.get_potential_energy()); fv = np.asarray(at.get_forces())
            ec, fc = pair_corr_gated(at, rho, args.A, args.b, args.ra, args.rb, args.power)
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
