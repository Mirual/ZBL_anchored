#!/usr/bin/env python3
"""Explainer figure: WHERE (in r) the physics anchor switches on.

Single panel: schematic dimer pair-interaction V(r) on a log scale with
three curves (true physics / vanilla softened MLIP / ZBL-anchored),
zone tints (analytic core / coverage gap / training domain), and a thin
gate-weight strip w(r) below.  Style matches make_method_scheme.py.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- palette (poster) -------------------------------------------------------
TEAL = "#17706b"
TEAL_DK = "#0e4f4b"
GREY_BG = "#f4f4f2"
GREY_BD = "#c9c9c4"
TEAL_BG = "#e9f2f0"
TEAL_BD = "#9cc4bf"
BOX_BD = "#5a5a5a"
INK = "#1c1c1c"
SUB = "#444444"
ORANGE = "#b96a12"
ORANGE_BG = "#fdf6ec"
RED = "#b03a2e"
GREEN = "#1e8449"
GREY_MLIP = "#8a8a85"

MONO = {"family": "DejaVu Sans Mono"}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---- schematic curves -------------------------------------------------------
r = np.linspace(0.1, 3.5, 900)

# 1) true short-range physics: screened-Coulomb-like wall, ~keV at r -> 0.2
V_true = 400.0 * np.exp(-r / 0.3) / r

# 2) vanilla MLIP: log-space dip (up to ~10x too low) inside 0.4..2.1 A,
#    rejoins the wall for r < ~0.4 (built-in ZBL) and V_true for r >= 2.2
dip = np.log(10.0) * sig((r - 0.42) / 0.07) * sig((2.05 - r) / 0.18)
V_van = V_true * np.exp(-dip)

# 3) ZBL-anchored: vanilla + w * [V_true - V_van]_+  with an r-driven gate
w = sig((1.95 - r) / 0.09)
V_anc = V_van + w * np.maximum(V_true - V_van, 0.0)

# ---- figure -----------------------------------------------------------------
FW, FH = 11.5, 7.0
fig, (ax, axw) = plt.subplots(
    2, 1, figsize=(FW, FH), dpi=220, sharex=True,
    gridspec_kw=dict(height_ratios=[4.4, 1.0], hspace=0.09))
fig.subplots_adjust(left=0.085, right=0.985, top=0.975, bottom=0.155)

XMIN, XMAX = 0.1, 3.5
YMIN, YMAX = 5e-4, 5e4
ax.set_xlim(XMIN, XMAX)
ax.set_yscale("log")
ax.set_ylim(YMIN, YMAX)

# ---- zone tints (both axes) -------------------------------------------------
for a in (ax, axw):
    a.axvspan(XMIN, 0.3, color="#e6e6e2", zorder=0)
    a.axvspan(0.3, 2.0, color=RED, alpha=0.055, zorder=0)
    a.axvspan(2.2, XMAX, color=GREEN, alpha=0.07, zorder=0)

# zone top labels
ax.text(0.205, 1.3e-3, "analytic ZBL core", rotation=90, ha="center",
        va="bottom", fontsize=10.5, color=SUB)
ax.text(1.15, 2.9e4, "coverage gap 0.5–2 Å\n(compressed but bonded)",
        ha="center", va="top", fontsize=12.5, color=RED, fontweight="bold",
        linespacing=1.35)
ax.text(2.85, 2.9e4, "training domain\n(near equilibrium)",
        ha="center", va="top", fontsize=12.5, color=GREEN, fontweight="bold",
        linespacing=1.35)

# ---- the problem: shaded gap between true physics and vanilla ---------------
ax.fill_between(r, V_van, V_true, where=(V_true > V_van) & (r > 0.38) & (r < 2.15),
                color=RED, alpha=0.18, lw=0, zorder=2)

# ---- curves -----------------------------------------------------------------
ax.plot(r, V_true, color=INK, lw=3.6, zorder=3, solid_capstyle="round",
        label="true short-range physics (ZBL / DFT dimer)")
ax.plot(r, V_van, color=GREY_MLIP, lw=2.6, ls=(0, (5, 2.4)), zorder=4,
        label="vanilla foundation MLIP (softened)")
ax.plot(r, V_anc, color=TEAL, lw=2.3, zorder=5, solid_capstyle="round",
        label="ZBL-anchored")

# ---- annotations ------------------------------------------------------------
# problem label, below the curves, arrow into the shaded wedge
ax.annotate("missing repulsion — GNN extrapolates",
            xy=(0.95, 2.2), xytext=(1.22, 0.02),
            fontsize=11.5, color=RED, fontweight="bold",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                            shrinkA=6, shrinkB=2, mutation_scale=16))

# gate-open annotation: anchored curve follows the wall
ax.annotate("w → 1:  + w·[V_ZBL − V_dimer]₊",
            xy=(0.52, 190.0), xytext=(0.72, 2.6e3),
            fontsize=12, color=TEAL_DK, family=MONO["family"], fontweight="bold",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.8,
                            shrinkA=4, shrinkB=3, mutation_scale=16))

# gate-closed annotation: bit-identical tail
ax.annotate("w = 0 → bit-identical",
            xy=(2.72, 0.017), xytext=(2.32, 0.9),
            fontsize=12, color=TEAL_DK, family=MONO["family"], fontweight="bold",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.8,
                            shrinkA=4, shrinkB=4, mutation_scale=16))

# ---- axes cosmetics ---------------------------------------------------------
ax.set_ylabel("pair energy (schematic, log scale)", fontsize=12.5, color=INK)
ax.set_yticks([1e-3, 1e-1, 1e1, 1e3])
ax.set_yticklabels(["1 meV", "0.1 eV", "10 eV", "1 keV"], fontsize=11)
ax.tick_params(axis="both", labelsize=11, colors=SUB, length=3.5)
ax.minorticks_off()
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GREY_BD)

leg = ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.775),
                frameon=False, fontsize=11.5, handlelength=2.5,
                borderaxespad=0.2, labelcolor=INK)

# ---- gate strip -------------------------------------------------------------
axw.plot(r, w, color=TEAL, lw=2.6, solid_capstyle="round", zorder=3)
axw.fill_between(r, 0, w, color=TEAL, alpha=0.14, lw=0, zorder=2)
axw.set_ylim(-0.12, 1.22)
axw.set_yticks([0, 1])
axw.set_yticklabels(["0", "1"], fontsize=11)
axw.set_ylabel("gate w", fontsize=12.5, color=INK)
axw.set_xlabel("interatomic distance r (Å)", fontsize=12.5, color=INK)
axw.set_xticks(np.arange(0.5, 3.51, 0.5))
axw.tick_params(axis="both", labelsize=11, colors=SUB, length=3.5)
for s in ("top", "right"):
    axw.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axw.spines[s].set_color(GREY_BD)

# ---- footnote ---------------------------------------------------------------
fig.text(0.5, 0.018,
         "The gate is driven by descriptor novelty ρ, not by r — "
         "for close contacts they coincide; on near-equilibrium structures "
         "w ≡ 0 and the output is exactly vanilla.",
         ha="center", va="bottom", fontsize=10, color=SUB)

out = sys.argv[1] if len(sys.argv) > 1 else "anchor_zones.png"
fig.savefig(out, dpi=220, facecolor="white")
print("saved", out)
