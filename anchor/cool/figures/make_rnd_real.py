"""REAL version of guide_rnd_mechanism (which was a 1D schematic) — now on OUR data.
On real 256-dimensional MACE-MH-0 descriptors:
  (A) predictor output vs target output component-wise: on the baseline (MPtrj) points lie on the diagonal
      (predictor has learned target), on keep (compressed/OOD) they move off the diagonal → this is novelty.
  (B) per-atom novelty ρ=‖pred−target‖² by dataset: baseline ≈0, keep with a long tail.

How it was computed (methodology):
  - datasets: keep_test (VASP strongly distorted/compressed perovskites 26_02_5and8), u200_test (clean
    distorted user-target), MPtrj-10k (foundation baseline). Files are in the preflight splits/.
  - per frame: descriptors = SECOND forward of MACE-MH-0 (calc.get_descriptors, 256/atom),
    normalise (desc−mu)/sd with values saved from rnd.pt, run through the FROZEN target-MLP and the trained
    predictor-MLP; novelty = mean over channels of the squared difference.
Env: gfnff-delta-mace (MACE-MH-0, float32, head mp_pbe_refit_add)."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
from mace.calculators import MACECalculator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rnd_anchor_predict import RNDGate, VAN

SP = os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight") + "/splits"
MPTRJ = os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz")
rng = np.random.default_rng(0)

calc = MACECalculator(model_paths=[VAN], device="cuda", default_dtype="float32", head="mp_pbe_refit_add")
gate = RNDGate("cuda")
# take the TUNED gate threshold (SelectiveNet) from pairphys_theta.json — the same one used in the deployed anchor
# and in rnd_explained.png; rnd.pt r_lo (p90 of baseline) is the raw calibration, not for the report.
_th = json.load(open(os.path.join(os.environ.get("ZBL_ANCHOR_RESULTS", "results"), "pairphys_theta.json")))
RLO = _th["r_lo"]
print(f"gate threshold (deployed): r_lo={RLO}  (rnd.pt raw r_lo={gate.r_lo})")


def outputs(xyz, n):
    """Returns (target_out [Natoms,demb], predictor_out, novelty [Natoms]) over n frames."""
    T, P, NOV = [], [], []
    for at in read(xyz, index=f":{n}"):
        d = np.asarray(calc.get_descriptors(at))
        x = torch.tensor((d - gate.mu) / gate.sd, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            t = gate.target(x).cpu().numpy(); p = gate.pred(x).cpu().numpy()
        T.append(t); P.append(p); NOV.append(((p - t) ** 2).mean(1))
    return np.concatenate(T), np.concatenate(P), np.concatenate(NOV)


SETS = [("MPtrj (baseline)", MPTRJ, 80, "#27ae60"),
        ("weakly distorted (target)", f"{SP}/u200_test.xyz", 22, "#2980b9"),
        ("distorted (compressed OOD)", f"{SP}/keep_test.xyz", 80, "#c0392b")]
res = {name: outputs(xyz, n) for name, xyz, n, _ in SETS}


def r2(t, p):
    ss_res = ((p - t) ** 2).sum(); ss_tot = ((t - t.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


for name, (T, P, NOV) in res.items():
    print(f"{name:20s}: R²(pred≈target)={r2(T.ravel(), P.ravel()):+.3f}  "
          f"novelty med={np.median(NOV):.4f} p99={np.percentile(NOV,99):.3f} max={NOV.max():.2f}  (n_at={len(NOV)})")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.4))

# ---- (A) predictor vs target component-wise (real descriptors) ----
# Two groups, as in the schematic: baseline (predictor≈target, on the diagonal) and keep (diverges more).
NPTS = 4000
Tk, Pk, _ = res["distorted (compressed OOD)"]
Tb, Pb, _ = res["MPtrj (baseline)"]
ik = rng.choice(Tk.size, size=min(NPTS, Tk.size), replace=False)
ib = rng.choice(Tb.size, size=min(NPTS, Tb.size), replace=False)
axA.scatter(Tk.ravel()[ik], Pk.ravel()[ik], s=8, alpha=0.30, color="#c0392b",
            label=f"distorted (compressed OOD): R²={r2(Tk.ravel(), Pk.ravel()):+.2f} — diverges")
axA.scatter(Tb.ravel()[ib], Pb.ravel()[ib], s=8, alpha=0.45, color="#27ae60",
            label=f"MPtrj (baseline): R²={r2(Tb.ravel(), Pb.ravel()):+.2f} — on the diagonal")
lim = [min(axA.get_xlim()[0], axA.get_ylim()[0]), max(axA.get_xlim()[1], axA.get_ylim()[1])]
axA.plot(lim, lim, "k--", lw=1.2, label="ideal predictor = target")
axA.set_xlim(lim); axA.set_ylim(lim)
axA.set_xlabel("TARGET network output (frozen), per channel"); axA.set_ylabel("PREDICTOR network output (trained on baseline)")
axA.set_title("(A) Real data: predictor ≈ target on the baseline (green on the diagonal),\n"
              "diverges more on distorted (red) — especially the tail of compressed contacts", fontsize=10.5, fontweight="bold")
axA.legend(fontsize=8.5, loc="upper left"); axA.grid(alpha=0.25)

# ---- (B) novelty ρ by dataset (real) ----
names = [n for n, *_ in SETS]; short = [n.split(" (")[0] for n in names]; cols = [c for *_, c in SETS]
bp = axB.boxplot([res[n][2] for n in names], tick_labels=short, showfliers=True, whis=(1, 99),
                 flierprops=dict(marker=".", ms=2, alpha=0.3, mec="#888"),
                 medianprops=dict(color="k", lw=1.6), patch_artist=True)
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.55)
axB.axhline(RLO, color="k", ls="--", lw=1.1); axB.text(0.6, RLO, " r_lo (gate threshold)", fontsize=8, va="bottom")
axB.set_yscale("log"); axB.set_ylabel("novelty ρ = ‖pred − target‖² (log)")
for i, n in enumerate(names):
    axB.annotate(f"med {np.median(res[n][2]):.2g}", (i + 1, np.median(res[n][2])), fontsize=8, ha="center", va="bottom")
axB.set_title("(B) Real novelty ρ: baseline ≈0 (familiar),\ndistorted spikes up (unfamiliar) → the gate opens",
              fontsize=10.5, fontweight="bold"); axB.grid(axis="y", alpha=0.3, which="both")

fig.suptitle("RND on OUR data (real-data analog of the guide_rnd_mechanism schematic)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rnd_real_mechanism.png")
fig.savefig(out, dpi=130)
print("saved", out)
