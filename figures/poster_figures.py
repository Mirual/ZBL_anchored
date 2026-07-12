# %% [markdown]
# # Poster figures — uncertainty-gated physics anchor (journal style, English)
#
# All panels are built from primary result files (raw prediction JSONs, ramp
# JSONs); the two gating signals (RND novelty and k-NN distance) are recomputed
# on the actual datasets and cached.
#
# Typography: every figure is sized so that 1 figure-inch = 100 poster pixels
# (its exact display width on the A0 sheet), and all figures share ONE font
# scale — the effective on-poster text size matches the poster body text.
#
# Run: `python poster_figures.py`
# (GPU needed once for the descriptor pass; cached in sig_cache.npz)

# %%
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("ZBL_ANCHOR_RESULTS", "results"))
POSTER = Path(__file__).resolve().parent / "poster"
DPA_WS = Path(os.environ.get("ZBL_DPA_WS", ""))  # TODO: set for your machine (DPA-3.1 comparison workspace)
M3G_WS = Path(os.environ.get("ZBL_M3GNET_WS", ""))
CHG_WS = Path(os.environ.get("ZBL_CHGNET_WS", ""))
PRE = Path(os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight"))
OUT = POSTER / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "anchor" / "scripts"))

# muted journal palette
C_VAN, C_ANC, C_ZBL, C_FT = "#8a97a0", "#1b7b84", "#c8781f", "#b3403c"
C_U200, C_DIST = "#4878a8", "#b3403c"

# one unified font scale for every panel (10.5 pt at 100 dpi ≈ 14.6 px on the
# poster — same as the poster body text)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 100, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.28, "axes.linewidth": 0.9,
})
ANN = 7.5  # annotation font size, one size everywhere

THETA = json.loads((RESULTS / "pairphys_theta.json").read_text())
R_LO, R_HI, POWER = THETA["r_lo"], THETA["r_hi"], THETA["power"]
print(f"deployed θ: r_lo={R_LO} r_hi={R_HI} power={POWER} λ={THETA['lam']}")


def smoothstep(x, lo, hi):
    t = np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


# %% [markdown]
# ## Gating signals on the real datasets (cached)
# Per-atom descriptors are computed once for MPtrj-300 (baseline) / u200_test
# (target) / keep_test — the distorted, compressed OOD set; both signals are
# evaluated on the same atoms: RND novelty (deployed gate) and k-NN distance
# in PCA-32 space (the rejected first-generation gate, `rho_reference.npz`).

# %%
CACHE = POSTER / "sig_cache.npz"
if CACHE.exists():
    z = np.load(CACHE)
    NOV = {k: z[f"nov_{k}"] for k in ("mptrj", "u200", "keep")}
    KNN = {k: z[f"knn_{k}"] for k in ("mptrj", "u200", "keep")}
    KNN_LO = float(z["knn_lo"])
    print("signals from cache:", {k: v.size for k, v in NOV.items()})
else:
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator
    from rnd_anchor_predict import RNDGate, VAN
    from rho_anchor_predict import Rho

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACECalculator(model_paths=[VAN], device=dev,
                          default_dtype="float32", head="mp_pbe_refit_add")
    gate = RNDGate(dev)
    rho = Rho()                       # kNN: PCA-32 bank of 40k MPtrj atoms, k=8
    KNN_LO = float(rho.r_lo)          # p90 of reference self-distances
    srcs = {
        "mptrj": (os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), "0:300"),
        "u200": (str(PRE / "splits" / "u200_test.xyz"), ":"),
        "keep": (str(PRE / "splits" / "keep_test.xyz"), ":"),
    }
    NOV, KNN = {}, {}
    for name, (path, sl) in srcs.items():
        frames = read(path, index=sl)
        desc = [np.asarray(calc.get_descriptors(a)) for a in frames]
        NOV[name] = np.concatenate([gate.novelty(d) for d in desc])
        KNN[name] = np.concatenate([rho.raw_dist(d) for d in desc])
        print(f"{name}: {len(frames)} frames, {NOV[name].size} atoms | "
              f"RND med={np.median(NOV[name]):.2e} max={NOV[name].max():.1f} | "
              f"kNN med={np.median(KNN[name]):.2f}")
    np.savez_compressed(CACHE, knn_lo=KNN_LO,
                        **{f"nov_{k}": v for k, v in NOV.items()},
                        **{f"knn_{k}": v for k, v in KNN.items()})

W = {k: smoothstep(v, R_LO, R_HI) ** POWER for k, v in NOV.items()}
rnd_above = {k: 100.0 * (v > R_LO).mean() for k, v in NOV.items()}
knn_above = {k: 100.0 * (v > KNN_LO).mean() for k, v in KNN.items()}
print("RND: % atoms above r_lo:", {k: round(v, 1) for k, v in rnd_above.items()})
print("kNN: % atoms above p90 threshold:", {k: round(v, 1) for k, v in knn_above.items()})
print(f"threshold margins: max mptrj rho={NOV['mptrj'].max():.4f} ({R_LO/NOV['mptrj'].max():.1f}x below r_lo), "
      f"max u200 rho={NOV['u200'].max():.4f} ({R_LO/NOV['u200'].max():.1f}x below)")

DSETS = (("mptrj", C_VAN, "MPtrj (baseline)"),
         ("u200", C_U200, "weakly distorted (target)"),
         ("keep", C_DIST, "distorted (compressed OOD)"))

# %% [markdown]
# ## Figure 1 — the gate on real data: RND novelty separates, k-NN distance does not
# Same atoms in both panels; one shared legend above the panels; explanatory
# detail lives in the caption, so nothing sits on top of the histograms.

# %%
HKW = dict(histtype="step", lw=2.0)
FKW = dict(histtype="stepfilled", alpha=0.18, lw=0)

fig, (ax, axk) = plt.subplots(1, 2, figsize=(11.72, 2.9), width_ratios=[1.12, 1])

# --- (a) RND novelty ---
bins = np.logspace(-5, 3, 90)
for key, color, label in DSETS:
    wts = np.full(NOV[key].size, 100.0 / NOV[key].size)
    ax.hist(NOV[key], bins=bins, weights=wts, color=color, **FKW)
    ax.hist(NOV[key], bins=bins, weights=wts, color=color, label=label, **HKW)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 1e3)
ax.set_ylim(9e-3, 300)
ax.set_xlabel(r"RND novelty $\rho$ (per atom)")
ax.set_ylabel("% of atoms")
ax.axvline(R_LO, color=C_ZBL, ls="--", lw=1.8)
ax.axvspan(R_LO, R_HI, color=C_ZBL, alpha=.10)
ax.set_title("(a)  RND novelty (deployed gate)", loc="left")

# no sigmoid overlay: the threshold alone separates the sets with zero errors;
# the shaded band is where the smooth ramp w(rho) turns on (force continuity)
ax.text(0.02, 0.97, "0% above threshold",
        transform=ax.transAxes, fontsize=ANN, color="#37474f",
        ha="left", va="top")
ax.text(np.sqrt(R_LO * R_HI), 1.0, "$w: 0 \\rightarrow 1$",
        fontsize=ANN - 1, color="#8a5512", ha="center", va="bottom")
ax.text(0.99, 0.96, "distorted tail: 5.1% $\\rightarrow$ correction",
        transform=ax.transAxes, fontsize=ANN, color=C_DIST,
        ha="right", va="top")

# --- (b) kNN distance ---
binsk = np.logspace(np.log10(0.05), np.log10(300), 80)
for key, color, label in DSETS:
    wts = np.full(KNN[key].size, 100.0 / KNN[key].size)
    axk.hist(KNN[key], bins=binsk, weights=wts, color=color, **FKW)
    axk.hist(KNN[key], bins=binsk, weights=wts, color=color, label=label, **HKW)
axk.set_xscale("log")
axk.set_yscale("log")
axk.set_xlim(0.05, 300)
axk.set_ylim(9e-3, 300)
axk.set_xlabel(r"$k$-NN distance in descriptor space (PCA-32, $k$=8)")
axk.axvline(KNN_LO, color=C_ZBL, ls="--", lw=1.8)
axk.set_title("(b)  $k$-NN distance (1st-gen gate, rejected)", loc="left")
axk.text(0.98, 0.98, "distributions overlap: threshold flags\n"
         f"{knn_above['mptrj']:.0f}% of baseline / {knn_above['u200']:.0f}% of target atoms",
         transform=axk.transAxes, fontsize=ANN, color="#37474f",
         ha="right", va="top")

# one shared legend above both panels (threshold explained once, for a and b)
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], color=C_ZBL, ls="--", lw=1.8))
labels.append("calibrated threshold")
fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False,
           bbox_to_anchor=(0.5, 1.005), columnspacing=1.4, handlelength=1.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "poster_gate.png", bbox_inches="tight")
plt.close(fig)
print("→ poster_gate.png")

# %% [markdown]
# ## Figure 2 — anchor vs fine-tuning (MACE-MH-0, forces)
# Sets: MPtrj (baseline) / full VASP test, 1573 frames, in-distribution /
# distorted compressed OOD (87 frames, `anchor_vs_ft_keep.json`).

# %%
avf = json.loads((RESULTS / "anchor_vs_ft.json").read_text())
avk = json.loads((RESULTS / "anchor_vs_ft_keep.json").read_text())
sets = [("MPtrj", avf["mptrj"]), ("VASP test", avf["all"]),
        ("distorted", avk["all"])]

fig, ax = plt.subplots(figsize=(3.37, 2.4))
x, wd = np.arange(len(sets)), 0.26
for j, (tag, label, color) in enumerate((("vanilla", "vanilla", C_VAN),
                                         ("anchor", "anchor", C_ANC),
                                         ("ft", "fine-tuned", C_FT))):
    vals = [s[1][tag]["F_R2"] for s in sets]
    clip = np.clip(vals, -1.15, 1.05)
    b = ax.bar(x + (j - 1) * wd, clip, wd, color=color, label=label)
    for bar, raw in zip(b, vals):
        if raw < -1.15:
            ax.text(bar.get_x() + wd / 2, -0.575, r"$-2\times10^{15}$", ha="center",
                    va="center", rotation=90, fontsize=7, color="#fff",
                    fontweight="bold")
        else:  # values inside the bars — outside they overlap on the narrow panel
            ax.text(bar.get_x() + wd / 2, raw - 0.03, f"{raw:.2f}", ha="center",
                    va="top", rotation=90, fontsize=ANN - 0.5, color="#fff",
                    fontweight="bold")
ax.axhline(0, color="#5a707c", lw=1.0)
ax.set_xticks(x, [s[0] for s in sets])
ax.set_ylabel(r"force $R^2$")
ax.set_ylim(-1.3, 1.12)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False,
           bbox_to_anchor=(0.5, 1.005), columnspacing=1.3, handlelength=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT / "poster_vs_ft.png", bbox_inches="tight")
plt.close(fig)
print("→ poster_vs_ft.png")

# %% [markdown]
# ## Figure 3 — cross-model result on the distorted set (single panel)
# One panel: force R² on the distorted compressed set for four MLIPs,
# vanilla / ungated ZBL / anchor. The price of removing the gate (baseline
# MPtrj R² of ungated ZBL) is printed under the ZBL bars. MACE numbers are
# recomputed from raw prediction files; others from compare_summary.json.


# %%
def f_r2_from_raw(path):
    d = json.loads(Path(path).read_text())
    ref = np.concatenate([np.asarray(f["F_ref"], float).ravel() for f in d["frames"]])
    pred = np.concatenate([np.asarray(f["F_pred"], float).ravel() for f in d["frames"]])
    ss = np.sum((ref - ref.mean()) ** 2)
    return 1.0 - np.sum((pred - ref) ** 2) / ss


mace_keep_v = f_r2_from_raw(RESULTS / "all_vanilla_keep.json")
mace_keep_a = f_r2_from_raw(RESULTS / "pairphys_keep.json")
mace_base_v = f_r2_from_raw(RESULTS / "all_vanilla_mptrj.json")
mace_base_a = f_r2_from_raw(RESULTS / "rndcal_mptrj.json")
print(f"MACE from raw: dist {mace_keep_v:.3f}→{mace_keep_a:.3f} | base {mace_base_v:.3f}→{mace_base_a:.3f}")

CS = {m: json.loads((w / "results" / "compare_summary.json").read_text())
      for m, w in (("DPA-3.1", DPA_WS), ("M3GNet", M3G_WS), ("CHGNet", CHG_WS))}
models = ["MACE-MH-0", "DPA-3.1", "M3GNet", "CHGNet"]
dist = {"vanilla": [mace_keep_v] + [CS[m]["keep_test/vanilla"]["F_r2"] for m in models[1:]],
        "ungated ZBL": [np.nan] + [CS[m]["keep_test/zbl"]["F_r2"] for m in models[1:]],
        "anchor": [mace_keep_a] + [CS[m]["keep_test/anchor"]["F_r2"] for m in models[1:]]}
zbl_base = [np.nan] + [CS[m]["mptrj/zbl"]["F_r2"] for m in models[1:]]

fig, ax = plt.subplots(figsize=(3.77, 2.4))
x, wd = np.arange(len(models)), 0.27
for j, (tag, color) in enumerate((("vanilla", C_VAN), ("ungated ZBL", C_ZBL),
                                  ("anchor", C_ANC))):
    v = np.asarray(dist[tag], float)
    b = ax.bar(x + (j - 1) * wd, np.where(np.isnan(v), 0, v), wd, color=color,
               label=tag)
    for xi, (raw, bar) in enumerate(zip(v, b)):
        if np.isnan(raw):
            ax.text(x[xi] + (j - 1) * wd, 0.03, "ZBL built-in", ha="center",
                    va="bottom", rotation=90, fontsize=7, color="#4b5d67")
        else:
            ax.text(bar.get_x() + wd / 2, max(raw, 0) + 0.035, f"{raw:.2f}",
                    ha="center", va="bottom", rotation=90, fontsize=ANN,
                    fontweight="bold")
# the cost of removing the gate (baseline R² of ungated ZBL) is carried by the figure caption, not the panel
ax.axhline(0, color="#5a707c", lw=1.0)
ax.set_xticks(x, ["MACE", "DPA-3.1", "M3GNet", "CHGNet"])
ax.tick_params(axis="x", pad=1)
ax.set_ylabel(r"force $R^2$ (distorted set)")
ax.set_ylim(-0.28, 1.32)
ax.set_yticks([-0.5, 0, 0.5, 1.0])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False,
           bbox_to_anchor=(0.5, 1.005), columnspacing=1.1, handlelength=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT / "poster_crossmodel.png", bbox_inches="tight")
plt.close(fig)
print("→ poster_crossmodel.png")

# %% [markdown]
# ## Figure 4 — compression to collapse (ramp; 40-structure statistics inline)

# %%
van = json.loads((RESULTS / "highP2_compressed_0_vanZBL.json").read_text())["ramp"]
pp = json.loads((RESULTS / "highP2_compressed_0_pairphysZBL.json").read_text())["ramp"]
ext = json.loads((RESULTS / "extended_highP.json").read_text())
n = ext["summary"]["n"]
vc, pc = ext["summary"]["vanilla_crashes"], ext["summary"]["pairphys_crashes"]

fig, ax = plt.subplots(figsize=(3.30, 2.4))
for ramp, color, label in ((van, C_VAN, "vanilla"), (pp, C_ANC, "anchor")):
    s = np.array([r["scale"] for r in ramp])
    de = np.array([r["e_per_atom"] for r in ramp]) - ramp[0]["e_per_atom"]
    crash = np.array([bool(r.get("crash")) for r in ramp])
    ax.plot(s[~crash], de[~crash], "-o", color=color, lw=2.2, ms=4.5, label=label)
    if crash.any():
        i = int(np.argmax(crash))
        ax.plot(s[i], 8, "x", ms=13, mew=3, color=C_FT, zorder=5)
        ax.text(s[i] - 0.018, 7.5, "collapse", fontsize=ANN, color=C_FT,
                ha="left", va="center")
ax.invert_xaxis()
ax.set_ylim(-8, 75)
ax.set_xlabel(r"compression $V/V_0$  (O$_2$Ti$_{12}$)")
ax.set_ylabel(r"$\Delta E$ per atom (eV)")
ax.legend(loc="upper left", borderpad=0.4)
ax.text(0.03, 0.58, f"40 structures:\ncollapses {vc}/{n} $\\rightarrow$ {pc}/{n}",
        transform=ax.transAxes, fontsize=ANN, color="#37474f", va="top")
fig.tight_layout()
fig.savefig(OUT / "poster_highP.png", bbox_inches="tight")
plt.close(fig)
print("→ poster_highP.png")

# %% [markdown]
# ## Figure 5 — molecular dynamics: dimer collision (NVE) without built-in ZBL
# `md_stability/results/collide_*.json`: closest approach r_min of an NVE
# head-on collision. When the built-in ZBL of MACE-MH-0 is removed, the gated
# Born–Mayer anchor restores (and exceeds) ZBL-grade repulsion. Thermal
# NVT/NVE (300–4000 K) and 2-MeV PKA cascades: zero collapses for every
# calculator; normal-set trajectories bit-identical to vanilla.

# %%
MDR = REPO / "anchor" / "md_stability" / "results"
cases = [("O-O", 60), ("Ti-O", 80), ("O-O", 200)]
coll = {f"{p}_{ke}": json.loads((MDR / f"collide_{p}_{ke}.json").read_text())["rmin"]
        for p, ke in cases}
variants = (("vanilla ZBL-on", C_VAN, "ZBL"),
            ("vanilla ZBL-off", C_FT, "no ZBL"),
            ("bornmayer ZBL-off", C_ANC, "+ anchor"))

from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ab initio reference: MP2 screening functions of Nordlund-Lehtola-Hobler,
# Phys. Rev. A 111, 032818 (2025); open data doi:10.5281/zenodo.14172632
# (local copy + provenance: poster/litdata/). V = 14.3996*Z1*Z2/r*phi;
# the MP2 curves carry a constant long-range reference offset -> zero at 10 A.
LIT = POSTER / "litdata"
ZNUM = {"O": 8, "Ti": 22}


def abinitio_turning(z1, z2, ke, d0=3.0):
    d = np.loadtxt(LIT / f"screening_mp2_{z1}_{z2}.dat")
    r, phi = d[1:, 0], d[1:, 1]
    V = 14.3996 * z1 * z2 / r * phi
    V -= V[np.argmin(abs(r - 10.0))]              # long-range MP2 offset
    target = ke + float(interp1d(r, V)(d0))       # collision starts at d0
    m = (r > 0.05) & (r < 2.0) & (V > 0)
    f = interp1d(np.log(r[m][::-1]), np.log(V[m][::-1]))
    return brentq(lambda rr: np.exp(f(np.log(rr))) - target,
                  r[m].min() * 1.01, r[m].max() * 0.99)


RT = []
for pr, ke in cases:
    z1, z2 = (ZNUM[t] for t in pr.split("-"))
    RT.append(abinitio_turning(z1, z2, ke))
    print(f"ab initio (MP2) turning point {pr} {ke} eV: {RT[-1]:.3f} A")

fig, ax = plt.subplots(figsize=(3.21, 2.2))
x, wd = np.arange(len(cases)), 0.27
for j, (tag, color, label) in enumerate(variants):
    vals = [coll[f"{p}_{ke}"][tag] for p, ke in cases]
    b = ax.bar(x + (j - 1) * wd, vals, wd, color=color, label=label)
    for xi, (bar, raw) in enumerate(zip(b, vals)):
        y0 = raw - 0.02
        if y0 >= RT[xi] > y0 - 0.22:      # the tick would cross the label — move below it
            y0 = RT[xi] - 0.02
        ax.text(bar.get_x() + wd / 2, y0, f"{raw:.2f}", ha="center",
                va="top", rotation=90, fontsize=ANN - 0.5, color="#fff",
                fontweight="bold")
        if xi == 0:  # variant names live inside the first-group bars — no legend box
            ax.text(bar.get_x() + wd / 2, 0.03, label, ha="center", va="bottom",
                    rotation=90, fontsize=9.5, color="#fff", fontweight="bold")
for xi, rturn in enumerate(RT):
    ax.hlines(rturn, xi - 0.42, xi + 0.42, colors="#16181a",
              linestyles=(0, (4, 3)), lw=1.6, zorder=4)
ax.text(0, 0.70, "ab initio\n(MP2) [8]", ha="center", va="bottom",
        fontsize=9, color="#16181a")
ax.set_xticks(x, [f"{p}\n{ke:g} eV" for p, ke in cases])
ax.set_ylabel(r"closest approach $r_{min}$ (Å)")
ax.set_ylim(0, 1.0)
fig.tight_layout()
fig.savefig(OUT / "poster_md.png", bbox_inches="tight")
plt.close(fig)
print("→ poster_md.png")

# %%
print("DONE:", sorted(p.name for p in OUT.glob("poster_*.png")))
