#!/usr/bin/env python3
"""WHY the short-range wall matters: radiation-damage cascade explainer.

Two panels for the README:
  (a) an alpha-recoil collision cascade in a waste-form ceramic lattice,
      with one highlighted close-approach pair (the ballistic phase probes
      r ~ 0.3-1.5 A, far inside any training data);
  (b) the causal chain from the V(r) wall to waste-form lifetime, and what
      breaks when an MLIP is softened there.

Style matches figures/make_method_scheme.py (palette, box style, large type).
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

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

MONO = {"family": "DejaVu Sans Mono"}

FW, FH = 11.5, 7.0
ASP = FW / FH  # multiply y-extents by this to get true circles
fig, ax = plt.subplots(figsize=(FW, FH), dpi=220)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)


def band(x, y, w, h, fc, ec, ls="-", lw=1.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.2",
                       fc=fc, ec=ec, lw=lw, ls=ls, mutation_aspect=ASP)
    ax.add_patch(p); return p


def arrow(x1, y1, x2, y2, c=BOX_BD, lw=2.4, ms=19, sa=0, sb=0):
    ax.annotate("", (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                shrinkA=sa, shrinkB=sb, mutation_scale=ms))


R = 1.05  # atom radius in x-units


def atom(x, y, fc="#e7e7e3", ec=GREY_BD, lw=1.1, z=3):
    ax.add_patch(Ellipse((x, y), 2 * R, 2 * R * ASP, fc=fc, ec=ec, lw=lw, zorder=z))


def vacancy(x, y):
    ax.add_patch(Ellipse((x, y), 2 * R, 2 * R * ASP, fc="none", ec="#8a8a85",
                         lw=1.3, ls=(0, (3.2, 2.4)), zorder=3))


def hit(x1, y1, x2, y2, lw=1.7):  # cascade arrow, atom-edge to atom-edge
    arrow(x1, y1, x2, y2, c=TEAL, lw=lw, ms=13, sa=9, sb=9)


# ============================ panel (a) =======================================
ax.text(1.5, 94.0, "(a)  collision cascade in a waste-form ceramic",
        fontsize=15, fontweight="bold", color=INK, va="center")

# hex-ish lattice: 8 rows x 9 cols
X0, Y0, DX, DY = 7.0, 20.0, 4.6, 7.4


def pos(i, j):
    return (X0 + j * DX + (i % 2) * DX / 2, Y0 + i * DY)


# original sites of displaced atoms (drawn as dashed vacancies instead)
S0 = pos(6, 2)            # PKA site
T1, T2 = pos(5, 4), pos(4, 3)
T3, T4 = pos(3, 6), pos(4, 6)
T5, T6 = pos(2, 3), pos(2, 7)
moved_sites = {S0, T1, T2, T3, T4, T5, T6}

for i in range(8):
    for j in range(9):
        p = pos(i, j)
        if p in moved_sites:
            vacancy(*p)
        else:
            atom(*p)

# displaced-atom positions (off-lattice)
P0 = (19.6, 61.0)          # PKA, keeps moving along recoil direction
D1 = (30.6, 54.4)
D2 = (21.6, 45.8)          # labelled interstitial
D3 = (39.4, 39.4)
D4 = (36.9, 46.4)
D5 = (20.0, 31.0)
CP = (40.0, 29.5)          # moving atom of the close pair
LT = pos(1, 7)             # lattice atom of the close pair (stays grey)

# incoming alpha-recoil
arrow(4.5, 82.0, S0[0] - 1.0, S0[1] + 1.7, c=ORANGE, lw=3.4, ms=22, sb=10)
ax.text(17.0, 84.0, "α-decay recoil nucleus", fontsize=11.5, color=ORANGE,
        fontweight="bold", va="center")
ax.text(17.0, 80.4, "10–100 keV", fontsize=11.5, color=ORANGE, va="center")

# cascade arrows (site-to-site, 3 generations)
hit(*P0, *T1)
hit(*P0, *T2)
hit(*D1, *T3)
hit(*D1, *T4)
hit(*D2, *T5)
hit(*D3, *T6)
hit(*T6, *CP)

# displaced atoms
for d in (D1, D2, D3, D4, D5, CP):
    atom(*d, fc=TEAL, ec=TEAL_DK, lw=1.2, z=4)
atom(*P0, fc=ORANGE, ec="#8a4e0d", lw=1.4, z=5)

# small labels
ax.text(17.2, 59.8, "PKA", fontsize=11, color=ORANGE, fontweight="bold",
        ha="right", va="center")
ax.text(23.2, 67.8, "vacancy", fontsize=10, color=SUB, ha="center", va="center")
ax.plot([20.2, 17.6], [67.4, 65.9], color="#8a8a85", lw=1.0)
ax.text(1.5, 46.0, "interstitial", fontsize=10, color=TEAL_DK, va="center")
ax.plot([9.0, 20.2], [46.0, 45.9], color=TEAL_BD, lw=1.0)

# close-approach pair
cpc = ((CP[0] + LT[0]) / 2, (CP[1] + LT[1]) / 2)
ax.add_patch(Ellipse(cpc, 5.6, 5.6 * ASP * 0.85, fc="none", ec=RED, lw=2.2,
                     zorder=6))
ax.plot([cpc[0] + 0.1, cpc[0] + 0.1], [16.6, cpc[1] - 4.2], color=RED, lw=1.1)
ax.text(38.2, 14.6, "close approach r ≈ 0.3–1.5 Å", fontsize=10.5, color=RED,
        fontweight="bold", ha="center", va="center")
ax.text(38.2, 11.4, "(ballistic phase)", fontsize=10.5, color=RED,
        ha="center", va="center")

# mini legend, bottom-left
atom(3.5, 14.0); ax.text(5.4, 14.0, "lattice atom", fontsize=9.5, color=SUB,
                         va="center")
atom(17.5, 14.0, fc=TEAL, ec=TEAL_DK)
ax.text(19.4, 14.0, "displaced", fontsize=9.5, color=SUB, va="center")
vacancy(3.5, 10.0); ax.text(5.4, 10.0, "vacancy site", fontsize=9.5, color=SUB,
                            va="center")

# ============================ panel (b) =======================================
ax.plot([49.6, 49.6], [5.0, 96.0], color=GREY_BD, lw=1.0)
ax.text(51.5, 94.0, "(b)  what the short-range wall controls",
        fontsize=15, fontweight="bold", color=INK, va="center")

CX, BW, BH = 64.5, 28.5, 11.5
CYs = (78.5, 63.5, 48.5, 33.5)
chain = [
    ("V(r) wall at 0.3–2 Å", "pair repulsion during collisions", True),
    ("threshold displacement energy E_d", "minimum recoil to create a defect",
     False),
    ("defects per cascade", "Frenkel pairs, replacement collisions", False),
    ("waste-form lifetime", "amorphization dose, swelling, leach rate", False),
]
for cy, (title, sub, mono) in zip(CYs, chain):
    band(CX - BW / 2, cy - BH / 2, BW, BH, "white", BOX_BD)
    kw = dict(ha="center", va="center", fontweight="bold")
    if mono:
        ax.text(CX, cy + 2.3, title, fontsize=14, color=TEAL,
                fontdict=dict(MONO), **kw)
    else:
        fs = 11.5 if len(title) > 26 else 13.5
        ax.text(CX, cy + 2.3, title, fontsize=fs, color=INK, **kw)
    ax.text(CX, cy - 2.7, sub, fontsize=10.5, color=SUB, ha="center",
            va="center")
for ya, yb in zip(CYs[:-1], CYs[1:]):
    arrow(CX, ya - BH / 2 - 0.8, CX, yb + BH / 2 + 0.8)

# red failure bracket
BRX = 80.6
ax.plot([BRX, BRX], [28.0, 84.0], color=RED, lw=2.0)
ax.plot([BRX - 1.1, BRX], [84.0, 84.0], color=RED, lw=2.0)
ax.plot([BRX - 1.1, BRX], [28.0, 28.0], color=RED, lw=2.0)
ax.text(82.6, 78.5, "MLIP softened here", fontsize=11, color=RED,
        fontweight="bold", va="center")
ax.text(82.6, 63.5, "→ E_d wrong", fontsize=10.5, color=RED, va="center")
ax.text(82.6, 48.5, "→ defect counts wrong", fontsize=10.5, color=RED,
        va="center")
ax.text(82.6, 33.5, "→ lifetime prediction\n    wrong", fontsize=10.5,
        color=RED, va="center", linespacing=1.4)

# teal closing line
ax.plot([56.0, 92.0], [21.4, 21.4], color=TEAL_BD, lw=1.2)
ax.text(74.0, 17.4, "anchor restores the wall; baseline MD", fontsize=11.5,
        color=TEAL, fontweight="bold", ha="center", va="center")
ax.text(74.0, 13.6, "(elasticity, WBM, 2-MeV cascades) unchanged",
        fontsize=11.5, color=TEAL, fontweight="bold", ha="center", va="center")

out = sys.argv[1] if len(sys.argv) > 1 else "radiation_cascade.png"
fig.savefig(out, dpi=220, facecolor="white")
print("saved", out)
