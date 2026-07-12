#!/usr/bin/env python3
"""RND gate (ported verbatim from MACE-anchor; model-agnostic — reads din/demb from rnd.pt).

novelty(atom) = ‖predictor(desc) − target(desc)‖² on normalized descriptors.
ρ = smoothstep(novelty; r_lo, r_hi) ∈ [0,1] — correction weight (0 in-distribution, 1 on OOD).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

RND = Path(__file__).resolve().parents[1] / "results" / "rnd.pt"


def mlp(din, demb):
    return nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, demb))


def smoothstep(x, lo, hi):
    """0 for x<lo, 1 for x>hi (grows with novelty) — correction gate."""
    t = np.clip((np.asarray(x) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class RNDGate:
    def __init__(self, dev, path=RND):
        z = torch.load(path, map_location=dev, weights_only=False)
        self.mu = z["mu"]; self.sd = z["sd"]; self.dev = dev
        self.r_lo, self.r_hi = z["r_lo"], z["r_hi"]
        self.target = mlp(z["din"], z["demb"]).to(dev); self.target.load_state_dict(z["target"]); self.target.eval()
        self.pred = mlp(z["din"], z["demb"]).to(dev); self.pred.load_state_dict(z["predictor"]); self.pred.eval()

    def novelty(self, desc):                       # desc [N,128] → per-atom novelty [N]
        x = torch.tensor((np.asarray(desc) - self.mu) / self.sd, device=self.dev, dtype=torch.float32)
        with torch.no_grad():
            return ((self.pred(x) - self.target(x)) ** 2).mean(1).cpu().numpy()
