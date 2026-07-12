#!/usr/bin/env python3
"""RND-gate (ported from RL-exploration, Burda 1810.12894): a learnable novelty signal instead of kNN-ρ.

A fixed random target network + a learnable predictor mimic it on per-atom MPtrj descriptors.
novelty(atom) = ‖predictor(desc) − target(desc)‖²  → high on environments far from the training foundation.
Theoretically ≈ a deep-ensemble variance in 1 forward (2602.19964), but without the non-bi-Lipschitz problem
of the raw distance: the predictor itself learns which feature directions matter.

Output: results/rnd.pt {target, predictor (state_dict), mu, sd, r_lo, r_hi}.
Calibration: r_lo=p90, r_hi=p99.5 novelty on held-out MPtrj atoms (as with the ρ-reference — for comparability).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from ase.io import read
from mace.calculators import MACECalculator

VAN = os.environ.get("ZBL_MACE_MH0", "/path/to/mace-mh-0.model")
MIXED = str(Path(os.environ.get("ZBL_MIXED_DATA", "/path/to/mixed_dataset/data")) / "mixed_train.xyz")
OUT = Path(__file__).resolve().parents[1] / "results" / "rnd.pt"
N_FRAMES = 1500
DIN, DEMB = 256, 128
EPOCHS, BS, LR = 40, 4096, 1e-3
SEED = 0


def mlp(din, demb):
    return nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, demb))


def descriptors(frames, calc):
    return np.concatenate([np.asarray(calc.get_descriptors(a, invariants_only=True)) for a in frames], 0)


def main():
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(model_paths=[VAN], device=dev, default_dtype="float32", head="mp_pbe_refit_add")
    frames = [a for a in read(MIXED, index=":") if a.info["REF_energy"] / len(a) < 50][:N_FRAMES]
    X = descriptors(frames, calc).astype(np.float32)
    print(f"reference: {len(frames)} MPtrj frames, {X.shape[0]} atoms, dim={X.shape[1]}")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    ntr = int(0.9 * len(X))
    Xtr, Xte = X[perm[:ntr]], X[perm[ntr:]]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6

    def norm(a): return torch.tensor((a - mu) / sd, device=dev)
    Xtr_t, Xte_t = norm(Xtr), norm(Xte)

    target = mlp(DIN, DEMB).to(dev)
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = mlp(DIN, DEMB).to(dev)
    opt = torch.optim.Adam(predictor.parameters(), lr=LR)

    with torch.no_grad():
        Ttr, Tte = target(Xtr_t), target(Xte_t)
    for ep in range(EPOCHS):
        predictor.train()
        idx = torch.randperm(len(Xtr_t), device=dev)
        tot = 0.0
        for k in range(0, len(idx), BS):
            b = idx[k:k + BS]
            opt.zero_grad()
            loss = ((predictor(Xtr_t[b]) - Ttr[b]) ** 2).mean()
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            predictor.eval()
            with torch.no_grad():
                te = ((predictor(Xte_t) - Tte) ** 2).mean().item()
            print(f"ep {ep:3d}  train {tot/len(Xtr_t):.4f}  test {te:.4f}")

    predictor.eval()
    with torch.no_grad():
        nov = ((predictor(Xte_t) - Tte) ** 2).mean(1).cpu().numpy()   # held-out per-atom novelty
    r_lo, r_hi = np.percentile(nov, [90, 99.5])
    print(f"novelty held-out MPtrj: med={np.median(nov):.4f}  r_lo(p90)={r_lo:.4f}  r_hi(p99.5)={r_hi:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(target=target.state_dict(), predictor=predictor.state_dict(),
                    mu=mu, sd=sd, r_lo=float(r_lo), r_hi=float(r_hi), din=DIN, demb=DEMB), OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
