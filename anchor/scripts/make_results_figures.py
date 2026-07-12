#!/usr/bin/env python3
"""Figures comparing held-out anchor vs vanilla on 4 datasets + overhead."""
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path("results"); FIG = Path("figures"); FIG.mkdir(exist_ok=True)

def met(f):
    fr = json.loads((R / f).read_text())["frames"]
    Er, Ep, nat, Fr, Fp = [], [], [], [], []
    for x in fr:
        if x["E_pred"] is None: continue
        Er.append(x["E_ref"]); Ep.append(x["E_pred"]); nat.append(x["n_atoms"])
        if x["F_ref"] and x["F_pred"]:
            Fr.append(np.array(x["F_ref"]).reshape(-1)); Fp.append(np.array(x["F_pred"]).reshape(-1))
    Er, Ep, nat = map(np.array, (Er, Ep, nat))
    Fr, Fp = np.concatenate(Fr), np.concatenate(Fp)
    return dict(eMAE=np.abs((Ep-Er)/nat).mean()*1000,
                eR2=1-((Er-Ep)**2).sum()/((Er-Er.mean())**2).sum(),
                fMAE=np.abs(Fp-Fr).mean(),
                fR2=1-((Fr-Fp)**2).sum()/((Fr-Fr.mean())**2).sum())

DS = [("u200\n(clean target)", "u200"), ("keep_test\n(compressed)", "keep"),
      ("keep_full 869\n(compressed)", "keepfull"), ("MPtrj-10k\n(base)", "mptrj")]
van = [met(f"all_vanilla_{t}.json") for _, t in DS]
anc = [met(f"all_anchor_{t}.json") for _, t in DS]
labels = [d for d, _ in DS]
x = np.arange(len(DS)); w = 0.38
CV, CA = "#5ب8def".replace("ب","b"), "#e8743b"

def diffmax(t):
    v = json.loads((R/f"all_vanilla_{t}.json").read_text())["frames"]
    a = json.loads((R/f"all_anchor_{t}.json").read_text())["frames"]
    m = []
    for vi, ai in zip(v, a):
        if vi["F_pred"] and ai["F_pred"]:
            d = np.abs(np.array(ai["F_pred"]) - np.array(vi["F_pred"]))
            m.append(d.max() if d.size else 0.0)
    return np.array(m)

# ===== FIG 1: 4 metrics, vanilla vs anchor =====
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
def bars(a, key, title, ylab, logy=False, clip=None):
    v = [m[key] for m in van]; n = [m[key] for m in anc]
    a.bar(x-w/2, v, w, label="vanilla", color=CV, edgecolor="k", lw=.5)
    a.bar(x+w/2, n, w, label="anchor", color=CA, edgecolor="k", lw=.5)
    a.set_title(title, fontweight="bold"); a.set_ylabel(ylab)
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=8)
    if logy: a.set_yscale("symlog")
    if clip: a.set_ylim(*clip)
    a.axhline(0, color="k", lw=.6); a.legend(fontsize=8); a.grid(axis="y", alpha=.3)
    for xi, (vv, nn) in enumerate(zip(v, n)):
        a.annotate(f"{nn:.2g}", (xi+w/2, nn), ha="center",
                   va="bottom" if nn >= 0 else "top", fontsize=7)
bars(ax[0,0], "fMAE", "Force MAE (↓ better)", "eV/Å", logy=True)
bars(ax[0,1], "fR2", "Force R² (↑ better)", "R²", clip=(-3.5, 1.1))
bars(ax[1,0], "eMAE", "Energy MAE (↓ better)", "meV/atom", logy=True)
bars(ax[1,1], "eR2", "Energy R² (↑ better)", "R²", clip=(-2.2, 1.1))
fig.suptitle("ρ-anchor vs vanilla — held-out, full datasets (MACE-MH-0)", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG/"heldout_metrics.png", dpi=130); plt.close(fig)

# ===== FIG 2: false-positives — max|ΔF| per frame =====
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for d, t in DS:
    m = diffmax(t); m = np.clip(m, 1e-3, None)
    ax[0].hist(np.log10(m), bins=40, alpha=.55, label=d.replace("\n"," "))
ax[0].set_xlabel("log₁₀ max|ΔF| per frame (eV/Å)"); ax[0].set_ylabel("frames")
ax[0].set_title("Where anchor changes forces\n(correction should be ~0 on base/clean)", fontweight="bold")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
frac = [(diffmax(t) > 1.0).mean()*100 for _, t in DS]
big = [(diffmax(t) > 1.0).sum() for _, t in DS]
b = ax[1].bar(x, frac, color=["#c0392b","#27ae60","#27ae60","#c0392b"], edgecolor="k")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
ax[1].set_ylabel("% frames with |ΔF|>1 eV/Å"); ax[1].set_title("Gate false positives", fontweight="bold")
for xi, (fr, bg) in enumerate(zip(frac, big)): ax[1].annotate(f"{bg} frames", (xi, fr), ha="center", va="bottom", fontsize=8)
ax[1].grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(FIG/"heldout_falsepos.png", dpi=130); plt.close(fig)

# ===== FIG 3: overhead =====
comp = ["vanilla\nE+F", "+descript.\n(2nd fwd)", "+ρ PCA+kNN", "+pair-corr"]
keep_t = [43.5, 40.5, 0.9, 0.8]; mp_t = [48.3, 35.5, 1.8, 0.9]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].bar(comp, keep_t, color=["#34495e","#c0392b","#e8743b","#f1c40f"], edgecolor="k")
ax[0].bar(comp, mp_t, alpha=0, edgecolor="none")
for i, (k, m) in enumerate(zip(keep_t, mp_t)): ax[0].annotate(f"{k:.0f}/{m:.0f}ms", (i, max(k,m)), ha="center", va="bottom", fontsize=8)
ax[0].set_ylabel("ms / structure"); ax[0].set_title("Component cost (keep/MPtrj)", fontweight="bold")
ax[0].tick_params(axis="x", labelsize=8); ax[0].grid(axis="y", alpha=.3)
modes = ["vanilla\n(1 fwd)", "anchor fused\n(1 fwd+ρ)", "anchor naive\n(2 fwd)"]
tk = [43.5, 43.5+0.9+0.8, 43.5+40.5+0.9+0.8]
bb = ax[1].bar(modes, tk, color=["#34495e","#27ae60","#c0392b"], edgecolor="k")
for i, v in enumerate(tk): ax[1].annotate(f"×{v/43.5:.2f}\n{v:.0f}ms", (i, v), ha="center", va="bottom", fontsize=9)
ax[1].set_ylabel("ms / structure"); ax[1].set_title("Full inference time (keep)", fontweight="bold")
ax[1].set_ylim(0, 100); ax[1].grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(FIG/"overhead.png", dpi=130); plt.close(fig)
print("saved:", *[p.name for p in FIG.glob("heldout*.png")], "overhead.png")
