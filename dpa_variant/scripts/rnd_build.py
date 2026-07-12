#!/usr/bin/env python3
"""RND gate on DPA-3.1 descriptors (port of MACE rnd_build.py; DIN=128 instead of 256).

A fixed random target network + a trainable predictor mimic it on per-atom DPA descriptors of the
MPtrj background. novelty = ‖pred−target‖² is high where DPA extrapolates. Output results/rnd.pt.
Calibration r_lo=p90, r_hi=p99.5 of novelty on held-out MPtrj atoms (as in MACE — for comparability).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from ase.io import read
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpa_common import RND_CORPUS, SPLITS, MPTRJ, DESC_DIM, compute_descriptors
from gate import mlp

OUT = Path(__file__).resolve().parents[1] / "results" / "rnd.pt"
N_FRAMES = 1000   # MPtrj background for RND; 1000 frames (~30k atoms) is enough, saves ~7 min of extract
DIN = DEMB = DESC_DIM          # 128
EPOCHS, BS, LR, SEED = 40, 4096, 1e-3, 0


def main():
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frames = [a for a in read(RND_CORPUS, index=":") if a.info["REF_energy"] / len(a) < 50][:N_FRAMES]
    X = np.concatenate(compute_descriptors(frames), 0).astype(np.float32)
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
        nov = ((predictor(Xte_t) - Tte) ** 2).mean(1).cpu().numpy()
    r_lo, r_hi = np.percentile(nov, [90, 99.5])
    print(f"\nnovelty held-out MPtrj: med={np.median(nov):.4f}  r_lo(p90)={r_lo:.4f}  r_hi(p99.5)={r_hi:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(target=target.state_dict(), predictor=predictor.state_dict(),
                    mu=mu, sd=sd, r_lo=float(r_lo), r_hi=float(r_hi), din=DIN, demb=DEMB), OUT)
    print(f"saved {OUT}")

    # sanity: does novelty separate OOD (keep/u200) from in-distribution (MPtrj)?
    from gate import RNDGate
    g = RNDGate(dev, OUT)
    srcs = {"keep": (f"{SPLITS}/keep_test.xyz", ":"),
            "u200": (f"{SPLITS}/u200_test.xyz", ":"),
            "mptrj": (MPTRJ, "0:300")}
    print("\nnovelty by dataset (expect keep/u200 ≫ mptrj):")
    for name, (path, sl) in srcs.items():
        nv = np.concatenate([g.novelty(d) for d in compute_descriptors(read(path, index=sl))])
        print(f"  {name:6s}: med={np.median(nv):.4f} p90={np.percentile(nv,90):.4f} max={nv.max():.4f}")


if __name__ == "__main__":
    main()
