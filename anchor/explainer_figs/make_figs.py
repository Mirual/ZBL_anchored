import os, sys, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
M3 = os.environ.get("ZBL_M3GNET_WS", "")
plt.rcParams["font.size"] = 11

# ---------- FIG 1: pipeline / what we add ----------
def box(ax, xy, w, h, text, fc, fs=9.5, ec="k"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", fontsize=fs, zorder=5)

def arrow(ax, a, b, color="k", text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.5,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+0.015, text, ha="center", fontsize=8, color=color)

fig, ax = plt.subplots(figsize=(13, 7.2)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
GREY="#dfe6e9"; BLUE="#aed6f1"; GREEN="#abebc6"; ORANGE="#fad7a0"; RED="#f5b7b1"
box(ax,(0.02,0.46),0.20,0.16,"Frozen\nfoundation-MLIP\n(MACE/DPA/M3GNet/CHGNet)\nNO fine-tuning",GREY,9)
box(ax,(0.30,0.74),0.20,0.12,"per-atom descriptors\n$d_i$  [64–256]",BLUE)
box(ax,(0.30,0.10),0.20,0.12,"E, F  (vanilla)",GREY)
box(ax,(0.56,0.74),0.22,0.13,"① RND novelty (from RL)\n$ρ_i=\\|pred(d_i)-target(d_i)\\|^2$",BLUE)
box(ax,(0.56,0.55),0.22,0.13,"② Gate\n$w_i=\\mathrm{smoothstep}(ρ_i;r_{lo},r_{hi})^p$\n0 on base · 1 on OOD",ORANGE)
box(ax,(0.30,0.30),0.26,0.16,"③ Per-pair physics (DimerCache)\n$ΔV_{ij}=[V_{ZBL}-V_{dimer}^{model}]_+ f_{cut}$\n+ core-ZBL below the grid",GREEN)
box(ax,(0.62,0.30),0.18,0.13,"④ Correction\n$ΔF=Σ\\,λ\\,w_{ij}\\,dΔV/dr$",ORANGE)
box(ax,(0.80,0.10),0.18,0.16,"$F_{anchor}=F_{van}+ΔF$\nwhere $w{=}0$: $=F_{van}$\n→ base INTACT",RED,9.5)
box(ax,(0.30,0.885),0.50,0.10,"⑤ SelectiveNet calibration:  $(r_{lo},λ)=\\arg\\min$ F-MAE(OOD)  s.t.  base untouched",GREEN,9)
# arrows
arrow(ax,(0.22,0.58),(0.30,0.80),text="descr")
arrow(ax,(0.22,0.52),(0.30,0.16),text="E,F")
arrow(ax,(0.50,0.80),(0.56,0.80))
arrow(ax,(0.67,0.74),(0.67,0.68))
arrow(ax,(0.56,0.40),(0.62,0.37))
arrow(ax,(0.67,0.55),(0.70,0.43))
arrow(ax,(0.71,0.30),(0.86,0.26))
arrow(ax,(0.40,0.10),(0.84,0.10),rad=-0.2)
ax.text(0.43,0.06,"$F_{vanilla}$",fontsize=8)
arrow(ax,(0.55,0.92),(0.40,0.46),color="#196f3d",rad=0.3)
ax.set_title("What we add: RND-gated physical anchor on top of a frozen foundation-MLIP",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_pipeline.png", dpi=130); plt.close(fig); print("fig1 ok")

# ---------- FIG 2: gate (smoothstep) ----------
def smoothstep(x, lo, hi, p=2):
    t = np.clip((x-lo)/(hi-lo), 0, 1); return (t*t*(3-2*t))**p
theta = json.load(open(f"{M3}/results/pairphys_theta.json"))
rlo, rhi, p = theta["r_lo"], theta["r_hi"], theta["power"]
x = np.logspace(-4, 0.6, 400)
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, smoothstep(x, rlo, rhi, p), lw=2.5, color="#e67e22")
ax.axvline(rlo, ls="--", color="#2980b9"); ax.axvline(rhi, ls="--", color="#2980b9")
ax.text(rlo, 1.05, f"$r_{{lo}}$={rlo:.3f}", color="#2980b9", ha="center", fontsize=9)
ax.text(rhi, 1.05, f"$r_{{hi}}$={rhi:.2f}", color="#2980b9", ha="center", fontsize=9)
ax.axvspan(1e-4, rlo, alpha=0.12, color="green"); ax.axvspan(rhi, 4, alpha=0.12, color="red")
ax.text(3e-4, 0.5, "in-distribution\n(base) → w≈0\ncorrection off", color="#196f3d", fontsize=9)
ax.text(0.8, 0.45, "OOD\n(compressed contacts)\nw≈1 → physics on", color="#922b21", fontsize=9, ha="center")
ax.set_xscale("log"); ax.set_xlabel("novelty  $ρ_i$  (RND)"); ax.set_ylabel("correction weight  $w_i$")
ax.set_title("② Gate: the correction turns on only for OOD atoms\n(smoothstep, θ from SelectiveNet — M3GNet)", fontweight="bold", fontsize=11)
ax.set_xlim(1e-4, 4); ax.set_ylim(0, 1.15); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_gate.png", dpi=130); plt.close(fig); print("fig2 ok")

# ---------- FIG 3: per-pair residual (real cached data) ----------
if M3:
    sys.path.insert(0, f"{M3}/scripts")
from pair_physics import zbl_V
cache = pickle.load(open(f"{M3}/results/dimer_m3gnet.pkl", "rb"))
key = (8, 22) if (8, 22) in cache else sorted(cache)[len(cache)//2]
e = cache[key]; zi, zj = key
r = e["r"]; dV = e["dV"]; rc = e["r_cut"]
vz = zbl_V(zi, zj, r)
vmodel = vz - dV   # f_cut≈1 at short r → V_model ≈ V_ZBL − residual
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(r, vz, lw=2.2, color="#2980b9", label="$V_{ZBL}$ (physical wall, diverges $\\sim Z_iZ_j/r$)")
ax.plot(r, np.clip(vmodel,0,None), lw=2, color="#7f8c8d", ls="--", label="$V^{model}_{dimer}$ (model's own dimer)")
ax.plot(r, dV, lw=2.6, color="#27ae60", label="$ΔV=[V_{ZBL}-V_{dimer}]_+\\,f_{cut}$ — WHAT WE ADD")
ax.axvline(rc, ls=":", color="k"); ax.text(rc, ax.get_ylim()[1]*0.6, f" $r_{{cut}}$={rc:.2f} Å\n(self-vanishing\nat the bond)", fontsize=8)
from ase.data import chemical_symbols
ax.set_title(f"③ Per-pair physics: we add ONLY the missing ZBL repulsion\n"
             f"(pair {chemical_symbols[zi]}–{chemical_symbols[zj]}, real DimerCache data)", fontweight="bold", fontsize=11)
ax.set_xlabel("r (Å)"); ax.set_ylabel("energy (eV)"); ax.set_xlim(0.3, rc+0.6)
ax.set_ylim(0, float(np.percentile(vz, 92))); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_residual.png", dpi=130); plt.close(fig); print("fig3 ok")

# ---------- FIG 4: novelty separation (recompute M3GNet subset) ----------
from m3gnet_common import load_model, compute_descriptors, SPLITS, MPTRJ
from gate import RNDGate
from ase.io import read
model, calc = load_model(); gate = RNDGate("cuda")
def nov_of(path, sl):
    fr = read(path, index=sl)
    return np.concatenate([gate.novelty(d) for d in compute_descriptors(fr, model=model)])
nk = nov_of(f"{SPLITS}/keep_test.xyz", ":")
nu = nov_of(f"{SPLITS}/u200_test.xyz", ":")
nm = nov_of(MPTRJ, "0:120")
fig, ax = plt.subplots(figsize=(8.5,5))
bins = np.logspace(-4.5, 0.6, 45)
for nv, c, lab in [(nm,"#7f8c8d","MPtrj (base)"),(nu,"#f39c12","u200 (clean target)"),(nk,"#c0392b","keep (compressed, OOD)")]:
    ax.hist(nv, bins=bins, alpha=0.55, color=c, label=f"{lab}  (max={nv.max():.3f})", density=True)
ax.axvline(rlo, ls="--", color="#2980b9"); ax.text(rlo,ax.get_ylim()[1]*0.9,f" $r_{{lo}}$",color="#2980b9")
ax.set_xscale("log"); ax.set_xlabel("novelty $ρ_i$ (per-atom, M3GNet)"); ax.set_ylabel("density")
ax.set_title("Why the gate works: keep has a SHARP novelty tail (compressed contacts),\n"
             "the base does not → the correction hits exactly the OOD atoms", fontweight="bold", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_novelty.png", dpi=130); plt.close(fig); print("fig4 ok")

# ---------- FIG 5: literature landscape ----------
fig, ax = plt.subplots(figsize=(9.5,7))
pts = [
 ("ZBL in training\n(NEP-ZBL, tabGAP, ACE-SR)", 0.12, 0.12, "#7f8c8d"),
 ("Fine-tuning on OOD\n(Deng 2024 — the \"proper\" fix)", 0.18, 0.42, "#7f8c8d"),
 ("bolt-on ZBL\n(lit.: \"ineffective\",\n\"distorts the PES\")", 0.88, 0.12, "#c0392b"),
 ("RND active-learning\n(PRL 132,167301)", 0.16, 0.9, "#7f8c8d"),
 ("learn-on-the-fly\n(uncertainty → QM calc)", 0.86, 0.78, "#7f8c8d"),
 ("multi-head committee UQ\n(JCP 2025, frozen MACE)", 0.78, 0.55, "#7f8c8d"),
]
for t,x,y,c in pts:
    ax.scatter([x],[y],s=120,color=c,zorder=5,edgecolor="k")
    ax.annotate(t,(x,y),fontsize=8.3,ha="center",va="bottom" if y<0.8 else "top",
                xytext=(0,8 if y<0.8 else -8),textcoords="offset points")
ax.scatter([0.9],[0.9],s=420,marker="*",color="#27ae60",zorder=6,edgecolor="k",lw=1.5)
ax.annotate("OUR anchor\ninference + uncertainty-GATED physics,\nno fine-tuning, base intact",
            (0.9,0.9),fontsize=9.5,fontweight="bold",ha="center",va="top",
            xytext=(0,-14),textcoords="offset points",color="#196f3d")
ax.add_patch(plt.Rectangle((0.62,0.62),0.36,0.36,fill=True,alpha=0.08,color="green"))
ax.text(0.8,0.985,"open niche",fontsize=9,color="#196f3d",ha="center",style="italic")
ax.set_xlabel("When the physics is applied  →  inference (post-hoc)", fontsize=10)
ax.set_ylabel("What turns it on  →  uncertainty-gated", fontsize=10)
ax.set_xticks([0.1,0.9]); ax.set_xticklabels(["training","inference"])
ax.set_yticks([0.1,0.9]); ax.set_yticklabels(["always","by uncertainty"])
ax.set_xlim(0,1.05); ax.set_ylim(0,1.08); ax.grid(alpha=0.3)
ax.set_title("Map of related work: every ingredient is published,\nbut the combination (top-right corner) is ours", fontweight="bold", fontsize=11.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_landscape.png", dpi=130); plt.close(fig); print("fig5 ok")
print("ALL FIGS DONE")
