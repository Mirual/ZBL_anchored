"""Distribution of FORCES by value (histograms, not the mean) across our datasets.
Shows the SHAPE of the |F| per-atom distribution, not summary statistics:
  (A) density (log-x) — where the mass of each dataset sits;
  (B) count in log-y (log-x) — the TAIL is visible: keep stretches to ~25000 eV/Å, the baseline is cut off at the bottom.
Datasets: keep (VASP compressed/distorted perovskites 26_02_5and8) · u200 (clean distorted user-target) ·
cleanmix (clean user) · MPtrj (foundation baseline). Values are REF (VASP/foundation), no model.
Env: any with ASE (xyz reading only)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read

SP = os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight") + "/splits"
SETS = [
    ("distorted (compressed OOD)", f"{SP}/keep_test.xyz", ":", "#c0392b"),
    ("weakly distorted (target)", f"{SP}/u200_test.xyz", ":", "#2980b9"),
    ("cleanmix (clean user)", f"{SP}/cleanmix_train.xyz", ":300", "#e67e22"),
    ("MPtrj (baseline)", os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz"), ":300", "#27ae60"),
]


def fmag(xyz, idx):
    out = []
    for at in read(xyz, index=idx):
        F = at.arrays.get("REF_forces", at.arrays.get("forces"))
        if F is not None:
            out.append(np.linalg.norm(np.asarray(F, float), axis=1))
    return np.concatenate(out) if out else np.array([])


data = {name: fmag(xyz, idx) for name, xyz, idx, *_ in SETS}
for name, F in data.items():
    print(f"{name:26s}: n={F.size:6d}  med={np.median(F):8.3f}  p50={np.percentile(F,50):8.3f}  "
          f"p90={np.percentile(F,90):9.2f}  p99={np.percentile(F,99):9.1f}  max={F.max():9.0f} eV/Å")

bins = np.logspace(-4, 5, 70)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2))

for name, _, _, c in SETS:
    F = np.clip(data[name], 1e-4, None)
    # (A) fraction of dataset atoms per bin — shapes are comparable across different dataset sizes
    axA.hist(F, bins=bins, weights=np.full(F.size, 1.0 / F.size), histtype="stepfilled",
             alpha=0.5, color=c, label=name)
    # (B) raw count — the tail is visible
    axB.hist(F, bins=bins, density=False, histtype="stepfilled", alpha=0.45, color=c, label=name)

for ax in (axA, axB):
    ax.set_xscale("log"); ax.set_xlabel("|F| per atom, eV/Å (log)")
    ax.grid(alpha=0.3, which="both")
    for name, _, _, c in SETS:
        ax.axvline(np.median(data[name]), color=c, ls=":", lw=1.3, alpha=0.9)

axA.set_ylabel("fraction of dataset atoms per bin")
axA.set_title("(A) Distribution of |F| by value (each dataset normalized to itself):\n"
              "keep is shifted 2–3 orders of magnitude to the right (dashed = medians)", fontsize=10.5, fontweight="bold")
axA.legend(fontsize=8.5, loc="upper left")

axB.set_yscale("log"); axB.set_ylabel("number of atoms (log)")
axB.set_title("(B) Same, count in log-y: the keep TAIL is visible up to ~25000 eV/Å\n"
              "(compressed contacts) — this regime is absent from the baseline/clean user", fontsize=10.5, fontweight="bold")
axB.legend(fontsize=8.5, loc="upper right")

fig.suptitle("Distribution of forces by value in our datasets (shape, not mean)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_histogram.png")
fig.savefig(out, dpi=130)
print("saved", out)
