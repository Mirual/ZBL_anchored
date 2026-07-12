#!/usr/bin/env python3
"""1D figure like the toy guide_rnd_mechanism, BUT on OUR data.

The toy drew predictor/target along an abstract 1D axis "atom environment". Here the same
2-panel layout, but:
  - X axis = REAL physical 1D coordinate: distance to the nearest neighbor d (Å) of each atom
    (small d = compressed/radiation contact = OOD; normal d = where the training base lived);
  - top: RND outputs projected onto PC1 of the target outputs (one scalar per atom) — target (gray) and
    predictor (purple) coincide in the trained zone and diverge at compressed contacts;
  - bottom: real novelty rho = ‖pred−target‖² along d; ≈0 in the trained zone, spikes at short d.
Green band = range of d covered by the MPtrj base (training distribution of RND).

Descriptors — a second MACE-MH-0 forward (256/atom), normalization (x−mu)/sd from rnd.pt, frozen target +
trained predictor. Env: gfnff-delta-mace.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ase.io import read  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WS + "/scripts")
from rnd_anchor_predict import RNDGate, VAN  # noqa: E402

SP = os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight") + "/splits"
MPTRJ = os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz")

calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
gate = RNDGate("cuda")
RLO = json.load(open(WS + "/results/pairphys_theta.json"))["r_lo"]


def per_atom(xyz, n):
    """Return (d_nn[Nat], target[Nat,demb], predictor[Nat,demb], novelty[Nat])."""
    D, T, P, NOV = [], [], [], []
    for at in read(xyz, index=f":{n}"):
        desc = np.asarray(calc.get_descriptors(at))
        x = torch.tensor((desc - gate.mu) / gate.sd, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            t = gate.target(x).cpu().numpy()
            p = gate.pred(x).cpu().numpy()
        i_idx, dist = neighbor_list("id", at, 4.0)
        dmin = np.full(len(at), np.inf)
        np.minimum.at(dmin, i_idx, dist)
        dmin[~np.isfinite(dmin)] = np.nan
        D.append(dmin); T.append(t); P.append(p); NOV.append(((p - t) ** 2).mean(1))
    return np.concatenate(D), np.vstack(T), np.vstack(P), np.concatenate(NOV)


dB, TB, PB, NB = per_atom(MPTRJ, 120)
dK, TK, PK, NK = per_atom(SP + "/keep_test.xyz", 87)

d = np.concatenate([dB, dK])
nov = np.concatenate([NB, NK])
ok = np.isfinite(d)
d, nov = d[ok], nov[ok]

DK = np.nanpercentile(dB, 1)   # boundary of the "familiar": to the left the base almost never went
print(f"known boundary DK={DK:.2f} Å   r_lo={RLO}")
print(f"atoms: base={np.isfinite(dB).sum()} keep={np.isfinite(dK).sum()}")


def binned(xv, yv, edges, stat):
    idx = np.digitize(xv, edges)
    xs, ys = [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() >= 3:
            xs.append(0.5 * (edges[b - 1] + edges[b])); ys.append(stat(yv[m]))
    return np.array(xs), np.array(ys)


edges = np.linspace(0.3, 3.5, 26)
XMIN, XMAX = 0.3, 3.5
RED, GREEN = "#e74c3c", "#27ae60"

print("\n  d-bin   n   med_rho   gate_open_%")
idx = np.digitize(d, edges)
for b in range(1, len(edges)):
    m = idx == b
    if m.sum() >= 3:
        print(f"  {0.5*(edges[b-1]+edges[b]):4.2f}  {m.sum():4d}  {np.median(nov[m]):.4f}  {100*np.mean(nov[m]>RLO):5.1f}")

fig, ax = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1]))

# zones: left (small d) UNFAMILIAR, right FAMILIAR
for a in ax:
    a.axvspan(XMIN, DK, color=RED, alpha=0.09, lw=0)
    a.axvspan(DK, XMAX, color=GREEN, alpha=0.09, lw=0)
    a.axvline(DK, color="#555", ls="--", lw=1.2)

# ---- TOP: how much the model does NOT know the environment (novelty ρ) ----
# points colored BY SET: distorted (compressed OOD) vs MPtrj (baseline)
mB, mK = np.isfinite(dB), np.isfinite(dK)
ax[0].scatter(dK[mK], np.clip(NK[mK], 1e-5, None), s=12, alpha=0.30, color="#c0392b",
              zorder=3, label="distorted (compressed OOD)")
ax[0].scatter(dB[mB], np.clip(NB[mB], 1e-5, None), s=12, alpha=0.30, color="#1e8449",
              zorder=2, label="MPtrj (baseline)")
xm, ym = binned(d, nov, edges, np.median)
ax[0].plot(xm, np.clip(ym, 1e-5, None), color="#154360", lw=3.2, zorder=5,
           label="median ρ (all atoms in the d bin)")
ax[0].axhline(RLO, color="k", ls=":", lw=1.6, zorder=3)
ax[0].text(XMAX - 0.05, RLO * 1.5, f"threshold: above it we ENABLE the correction (r_lo={RLO})",
           color="k", fontsize=9, ha="right", va="bottom")
ax[0].set_yscale("log")
ax[0].set_ylabel("novelty ρ\n(0 = model knows,  large = does NOT know)", fontsize=10)
ax[0].set_title("DETECTOR SIGNAL:  ρ = how far the student-predictor diverged from the reference-target",
                fontsize=12, fontweight="bold")
ax[0].legend(loc="center right", fontsize=9.5, framealpha=0.95)
ax[0].grid(alpha=0.25, which="both")
ax[0].text(0.62, ym.max() * 0.5 if len(ym) else 1, "✗ UNFAMILIAR\ncompressed contacts\nρ is LARGE",
           color=RED, fontsize=11, fontweight="bold", ha="center", va="center")
ax[0].text(2.2, RLO * 0.06, "✓ FAMILIAR (model was trained here) — ρ ≈ 0",
           color="#1e8449", fontsize=11, fontweight="bold", ha="center")

# ---- BOTTOM: what anchor does about it (fraction of enabled correction) ----
xg, yg = binned(d, (nov > RLO).astype(float) * 100, edges, np.mean)
ax[1].plot(xg, yg, color="#8e44ad", lw=3.2, marker="o", ms=4,
           label="fraction of atoms where the gate is OPEN (correction added)")
ax[1].set_ylim(-5, 105)
ax[1].set_ylabel("correction added\nto fraction of atoms, %", fontsize=10)
ax[1].set_xlabel("distance to nearest neighbor d, Å      ←  compressed contact (radiation impact)   |   normal bond  →",
                 fontsize=10.5)
ax[1].set_title("WHAT WE DO:  the gate opens ONLY at compressed contacts → the base (right side) is left untouched",
                fontsize=12, fontweight="bold")
ax[1].legend(loc="center right", fontsize=9.5, framealpha=0.95)
ax[1].grid(alpha=0.25)
ax[1].text(0.62, 60, "ON\nadd\nphysics", color="#6c3483", fontsize=11, fontweight="bold", ha="center")
ax[1].text(2.2, 12, "OFF — we touch nothing (base intact)", color="#6c3483", fontsize=11, fontweight="bold", ha="center")
ax[1].set_xlim(XMIN, XMAX)

# boundary label
ax[0].annotate("boundary: to the left the model\nwas barely trained", xy=(DK, ax[0].get_ylim()[1] * 0.2),
               xytext=(DK + 0.25, ax[0].get_ylim()[1] * 0.4), fontsize=8.5, color="#555",
               arrowprops=dict(arrowstyle="->", color="#555"))

fig.suptitle("How RND decides WHERE to fix — on OUR data (X axis = real distance to neighbor)",
             fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = WS + "/figures/rnd_1d_real.png"
fig.savefig(out, dpi=140)
import shutil
shutil.copy(out, WS + "/cool/figures/rnd_1d_real.png")
print("saved", out, "(+cool)")
