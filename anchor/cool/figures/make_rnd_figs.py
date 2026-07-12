"""3 figures for the RND section in HOW_IT_WORKS.md:
  (A) RND schematic (frozen target + trained predictor → novelty);
  (B) novelty histogram on REAL data (keep=OOD vs u200=target vs MPtrj=baseline) — separation;
  (C) gate curve w=smoothstep(novelty)^p, showing where keep/baseline sit.
Env: gfnff-delta-mace."""
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from ase.io import read
from mace.calculators import MACECalculator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rnd_anchor_predict import RNDGate, VAN
from anchor_predict import smoothstep

SP = os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight") + "/splits"
MPTRJ = os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz")
calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
gate = RNDGate("cuda")
th = json.load(open(os.path.join(os.environ.get("ZBL_ANCHOR_RESULTS", "results"), "pairphys_theta.json")))
rlo, rhi, pw = th["r_lo"], th["r_hi"], th.get("power", 2.0)
print(f"theta: r_lo={rlo}, r_hi={rhi}, power={pw}")


def novs(xyz, n):
    out = []
    for at in read(xyz, index=f":{n}"):
        out.append(gate.novelty(np.asarray(calc.get_descriptors(at))))
    return np.concatenate(out)


nk = novs(f"{SP}/keep_test.xyz", 80)
nu = novs(f"{SP}/u200_test.xyz", 22)
nm = novs(MPTRJ, 80)
fk, fu, fm = (nk > rlo).mean(), (nu > rlo).mean(), (nm > rlo).mean()   # fraction "gate open"
print(f"novelty med: keep={np.median(nk):.4f} u200={np.median(nu):.4f} mptrj={np.median(nm):.4f}; keep max={nk.max():.2f}")
print(f"fraction novelty>r_lo: keep={fk:.1%} u200={fu:.1%} mptrj={fm:.1%}; keep p90={np.percentile(nk,90):.3f} p99={np.percentile(nk,99):.2f}")

fig = plt.figure(figsize=(16, 4.7))

# ---- (A) RND schematic ----
ax = fig.add_subplot(1, 3, 1); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=fc, ec="k", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="bold")
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333"))
box(0.3, 4.2, 1.8, 1.6, "descriptor\nd ∈ ℝ²⁵⁶", "#ecf0f1")
box(3.4, 6.6, 3.2, 1.7, "TARGET-MLP\n(random, FROZEN)", "#d5d8dc")
box(3.4, 1.7, 3.2, 1.7, "PREDICTOR-MLP\n(trained on baseline)", "#aed6f1")
box(7.7, 4.2, 2.0, 1.6, "novelty\n‖pred−target‖²", "#f9e79f")
arrow(2.1, 5.4, 3.4, 7.3); arrow(2.1, 4.6, 3.4, 2.7)
arrow(6.6, 7.3, 7.7, 5.5); arrow(6.6, 2.7, 7.7, 4.6)
ax.text(5.0, 0.6, "predictor learns to mimic target ONLY on baseline (MPtrj) descriptors",
        ha="center", fontsize=7.8, style="italic", color="#555")
ax.set_title("(A) Random Network Distillation: trainable predictor\ncatches up to the random frozen target",
             fontsize=10, fontweight="bold")

# ---- (B) novelty histogram (real data) ----
ax2 = fig.add_subplot(1, 3, 2)
bins = np.logspace(-5, 3, 60)
for d, c, lbl in [(nm, "#27ae60", "MPtrj (baseline)"), (nu, "#2980b9", "weakly distorted (target)"), (nk, "#c0392b", "distorted (compressed OOD)")]:
    ax2.hist(np.clip(d, 1e-5, None), bins=bins, alpha=0.55, color=c, label=lbl, density=True)
ax2.axvline(rlo, color="k", ls="--", lw=1.3); ax2.text(rlo, ax2.get_ylim()[1] * 0.9, " r_lo", fontsize=9)
ax2.axvline(rhi, color="k", ls=":", lw=1.1); ax2.text(rhi, ax2.get_ylim()[1] * 0.75, " r_hi", fontsize=9)
ax2.set_xscale("log"); ax2.set_xlabel("novelty = ‖pred − target‖² (log)"); ax2.set_ylabel("density")
ax2.set_title("(B) novelty: separation in the TAIL (real data)\n"
              "distorted set has a long tail of compressed contacts up to ~400", fontsize=10, fontweight="bold")
ax2.text(0.97, 0.55, f"fraction novelty > r_lo\n(gate opens):\ndistorted {fk:.0%}\nweakly distorted {fu:.0%}\nbaseline {fm:.0%}",
         transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
         bbox=dict(boxstyle="round", fc="#fdf6e3", ec="k"))
ax2.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(0.97, 1.0))

# ---- (C) gate curve ----
ax3 = fig.add_subplot(1, 3, 3)
x = np.logspace(-5, 3, 400)
w = smoothstep(x, rlo, rhi) ** pw
ax3.plot(x, w, color="#8e44ad", lw=2.4)
ax3.axvspan(1e-5, rlo, color="#27ae60", alpha=0.12); ax3.axvspan(rhi, 1e3, color="#c0392b", alpha=0.10)
ax3.axvline(np.percentile(nm, 99), color="#196f3d", ls="-", lw=1.8)
ax3.text(np.percentile(nm, 99), 0.30, " baseline p99\n(all mass w≈0)", color="#196f3d", fontsize=7.3, rotation=90, va="center")
ax3.axvline(np.median(nk), color="#c0392b", ls=":", lw=1.4)
ax3.text(np.median(nk), 0.63, " distorted med\n(bulk, w≈0)", color="#c0392b", fontsize=7.3, rotation=90, va="center")
ax3.axvline(np.percentile(nk, 99), color="#c0392b", ls="-", lw=2.0)
ax3.text(np.percentile(nk, 99), 0.40, " distorted p99\n(compressed contacts,\nw≈1)", color="#c0392b", fontsize=7.3, rotation=90, va="center")
ax3.axvline(rlo, color="k", ls="--", lw=1); ax3.axvline(rhi, color="k", ls=":", lw=1)
ax3.set_xscale("log"); ax3.set_xlabel("novelty (log)"); ax3.set_ylabel("correction weight w")
ax3.set_ylim(-0.03, 1.05)
ax3.set_title("(C) Gate: w = smoothstep(novelty; r_lo, r_hi)^p\n"
              "baseline → w=0 (untouched); distorted (OOD) → w=1 (correction on)", fontsize=10, fontweight="bold")

fig.suptitle("RND-novelty gate: how anchor knows WHERE the model extrapolates (and only corrects there)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rnd_explained.png")
fig.savefig(out, dpi=130)
print("saved", out)
