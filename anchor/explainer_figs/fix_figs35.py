import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.data import chemical_symbols

OUT = os.path.dirname(os.path.abspath(__file__))
M3 = os.environ.get("ZBL_M3GNET_WS", "")
if M3:
    sys.path.insert(0, f"{M3}/scripts")
from pair_physics import zbl_V, _smoothstep
plt.rcParams["font.size"] = 11

# ============ FIG 3: annotated per-pair residual ============
cache = pickle.load(open(f"{M3}/results/dimer_m3gnet.pkl", "rb"))
key = (8, 22) if (8, 22) in cache else sorted(cache)[len(cache)//2]
e = cache[key]; zi, zj = key
r = e["r"]; dV = e["dV"]; rc = e["r_cut"]
vz = zbl_V(zi, zj, r)
vmodel = np.clip(vz - dV, 0, None)
fcut = _smoothstep(r, rc - 0.40, rc)

fig, ax = plt.subplots(figsize=(10, 6.3))
ax.plot(r, vz,     lw=2.4, color="#2980b9", label="① $V_{ZBL}$ — exact physics ONLY at short r")
ax.plot(r, vmodel, lw=2.0, color="#7f8c8d", ls="--", label="② $V^{model}_{dimer}$ — what the model outputs")
ax.plot(r, dV,     lw=3.2, color="#27ae60", label="③ $ΔV=[V_{ZBL}-V_{dimer}]_+\\,f_{cut}$ — what we ADD")
ax.fill_between(r, 1e-2, dV, color="#27ae60", alpha=0.10)
ax.set_yscale("log"); ax.set_ylim(1.0, max(vz.max(), dV.max())*1.7); ax.set_xlim(0.3, rc+0.55)
ax.axvline(rc, ls=":", color="k")

# problem A: gap = missing repulsion (between ① and ②)
ra = 0.52; ia = int(np.argmin(np.abs(r-ra)))
ax.annotate("", xy=(ra, vz[ia]), xytext=(ra, max(vmodel[ia],1)),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.6))
ax.annotate("PROBLEM A:\ngap ①−② =\nmodel is UNDER-repulsive\n(softening) → this is what we add",
            (ra, np.sqrt(vz[ia]*max(vmodel[ia],1))), color="#c0392b", fontsize=8.3,
            ha="left", va="center", xytext=(ra+0.12, 200), textcoords="data",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
# problem B: non-monotonicity of ΔV (artifact of the model's dimer curve)
ib = int(np.argmin(np.abs(r-0.82)))
ax.annotate("PROBLEM B:\ndip in ③ — artifact of\nthe model's dimer curve\n(its extrapolation is noisy)",
            (r[ib], dV[ib]), color="#7d3c98", fontsize=8.3, ha="center", va="top",
            xytext=(0.95, 7), textcoords="data",
            arrowprops=dict(arrowstyle="->", color="#7d3c98", lw=1))
# f_cut envelope
ax2 = ax.twinx(); ax2.plot(r, fcut, lw=1.4, color="#e67e22", alpha=0.7)
ax2.fill_between(r, 0, fcut, color="#e67e22", alpha=0.05); ax2.set_ylim(0, 1.05)
ax2.set_ylabel("$f_{cut}$ (cutoff envelope)", color="#e67e22", fontsize=9); ax2.tick_params(axis="y", labelcolor="#e67e22")
ax.annotate("PROBLEM C:\n$f_{cut}{\\to}0$ near the bond — we DON'T touch\na valid bond, but also DON'T fix\nintermediate-softening beyond $r_{cut}$",
            (rc, 1.6), color="#b9770e", fontsize=8.3, ha="right", va="bottom",
            xytext=(rc-0.05, 1.6), textcoords="data")
ax.text(rc+0.01, 60, f"$r_{{cut}}$={rc:.2f} Å", fontsize=8.5)
ax.set_title(f"③ Per-pair physics (pair {chemical_symbols[zi]}–{chemical_symbols[zj]}, real DimerCache data, M3GNet, log-Y)",
             fontweight="bold", fontsize=11)
ax.set_xlabel("r (Å)  —  distance within the pair"); ax.set_ylabel("energy (eV, log)")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_residual.png", dpi=130); plt.close(fig); print("fig3 ok")

# ============ FIG 5: landscape with numbered key (no overlap) ============
fig, ax = plt.subplots(figsize=(9.5, 8.6))
fig.subplots_adjust(bottom=0.30, top=0.92)
pts = [
 ("1", 0.12, 0.12, "#7f8c8d"),
 ("2", 0.20, 0.42, "#7f8c8d"),
 ("3", 0.88, 0.13, "#c0392b"),
 ("4", 0.16, 0.90, "#7f8c8d"),
 ("5", 0.86, 0.80, "#7f8c8d"),
 ("6", 0.74, 0.55, "#7f8c8d"),
]
ax.add_patch(plt.Rectangle((0.60,0.60),0.40,0.40,fill=True,alpha=0.08,color="green"))
ax.text(0.80,0.985,"open niche",fontsize=9,color="#196f3d",ha="center",style="italic")
for n,x,y,c in pts:
    ax.scatter([x],[y],s=260,color=c,zorder=5,edgecolor="k",lw=1.2)
    ax.text(x,y,n,ha="center",va="center",fontsize=10,fontweight="bold",color="white",zorder=6)
ax.scatter([0.90],[0.92],s=560,marker="*",color="#27ae60",zorder=6,edgecolor="k",lw=1.5)
ax.text(0.90,0.92,"★",ha="center",va="center",fontsize=11,color="white",zorder=7)
ax.set_xlabel("When the physics is applied  →  inference (post-hoc)", fontsize=10.5)
ax.set_ylabel("What triggers it  →  uncertainty-gated", fontsize=10.5)
ax.set_xticks([0.1,0.9]); ax.set_xticklabels(["training","inference"])
ax.set_yticks([0.1,0.9]); ax.set_yticklabels(["always","by uncertainty"])
ax.set_xlim(0,1.05); ax.set_ylim(0,1.08); ax.grid(alpha=0.3)
ax.set_title("Map of analogues: every ingredient has been published,\nbut the combination (★, top-right corner) is ours",
             fontweight="bold", fontsize=12)
key = [
 ("1", "ZBL in training (NEP-ZBL, tabGAP, ACE-SR)"),
 ("2", "Fine-tuning on OOD (Deng 2024 — the \"proper\" fix)"),
 ("3", "bolt-on ZBL post-hoc (lit.: \"ineffective\")"),
 ("4", "RND active-learning (PRL 132,167301 — data selection)"),
 ("5", "learn-on-the-fly (uncertainty → QM calculation)"),
 ("6", "committee UQ (JCP 2025, frozen MACE — error-bars)"),
 ("★", "OUR anchor: inference + uncertainty-GATED physics,"),
]
fig.text(0.04, 0.245, "Legend:", fontsize=10.5, fontweight="bold")
for i,(n,t) in enumerate(key):
    yy = 0.205 - i*0.029
    col = "#196f3d" if n=="★" else "k"
    fig.text(0.05, yy, f"{n} — {t}", fontsize=9.3, color=col,
             fontweight="bold" if n=="★" else "normal")
fig.text(0.072, 0.205-7*0.029, "no fine-tuning and no QM calls, the base model stays intact", fontsize=9.3, color="#196f3d")
fig.savefig(f"{OUT}/fig5_landscape.png", dpi=130); plt.close(fig); print("fig5 ok")
print("DONE")
