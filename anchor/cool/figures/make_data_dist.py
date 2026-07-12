"""Distribution of data by FORCES and ENERGIES across our datasets:
  keep (compressed/OOD) · u200 (clean user-target) · cleanmix (clean user) · MPtrj (foundation baseline).
(A) |F_i| per atom (log) — keep has a long tail of huge forces (compressed contacts) = why it is OOD.
(B) E per atom — user datasets are offset from MPtrj by ~589 eV/atom (E0-frame-gap) = why energy is not the anchor's target.
Env: deepmd_env (ASE reading only, no model)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read

SP = os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight") + "/splits"
SETS = [
    ("keep (compressed/OOD)", f"{SP}/keep_test.xyz", ":", "#c0392b"),
    ("u200 (user target)", f"{SP}/u200_test.xyz", ":", "#2980b9"),
    ("cleanmix (clean user)", f"{SP}/cleanmix_train.xyz", ":300", "#e67e22"),
    ("MPtrj (baseline)", os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), ":300", "#27ae60"),
]


def fe(xyz, idx):
    Fmag, Eat = [], []
    for at in read(xyz, index=idx):
        F = at.arrays.get("REF_forces", at.arrays.get("forces"))
        if F is not None:
            Fmag.append(np.linalg.norm(np.asarray(F, float), axis=1))
        E = at.info.get("REF_energy", at.info.get("energy"))
        if E is not None:
            Eat.append(float(E) / len(at))
    return np.concatenate(Fmag) if Fmag else np.array([]), np.array(Eat)


data = {name: fe(xyz, idx) for name, xyz, idx, _ in SETS}
for name, (F, E) in data.items():
    print(f"{name:26s}: |F| med={np.median(F):.2f} p99={np.percentile(F,99):.1f} max={F.max():.0f} eV/Å | "
          f"E/atom med={np.median(E):.1f} eV (n_at={len(F)})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
names = [n for n, *_ in SETS]
short = [n.split(" (")[0] for n in names]
colors = [c for *_, c in SETS]


def boxes(ax, arrays):
    bp = ax.boxplot(arrays, tick_labels=short, showfliers=True, whis=(1, 99),
                    flierprops=dict(marker=".", ms=2, alpha=0.25, mec="#888"),
                    medianprops=dict(color="k", lw=1.6), patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    return bp


# (A) forces |F| per atom — log-y box (range ~5 orders of magnitude)
boxes(ax1, [data[n][0] for n in names])
ax1.set_yscale("log"); ax1.set_ylabel("|F| per atom, eV/Å (log)")
for i, n in enumerate(names):
    ax1.annotate(f"med {np.median(data[n][0]):.2g}", (i + 1, np.median(data[n][0])),
                 fontsize=7.5, ha="center", va="bottom")
ax1.set_title("(A) Forces: keep is 2–3 orders of magnitude above baseline\n(compressed contacts, max ~25000 eV/Å) = OOD regime",
              fontsize=10.5, fontweight="bold"); ax1.grid(axis="y", alpha=0.3, which="both")

# (B) energies E per atom — box (keep is extremely high)
boxes(ax2, [data[n][1] for n in names])
ax2.set_ylabel("E per atom, eV")
for i, n in enumerate(names):
    ax2.annotate(f"{np.median(data[n][1]):+.0f}", (i + 1, np.median(data[n][1])),
                 fontsize=7.5, ha="center", va="bottom")
ax2.set_title("(B) Energies: keep is extreme (+558 eV/atom,\ncompression), clean user and baseline ~−6 eV/atom",
              fontsize=10.5, fontweight="bold"); ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Distribution of our datasets by forces and energies", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_distributions.png")
fig.savefig(out, dpi=130)
print("saved", out)
