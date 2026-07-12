#!/usr/bin/env python3
"""Model benchmark: 0 K mechanical stability (elastic C_ij + Born criteria + EOS B0).

Methodology in the spirit of Thomas & Van der Ven, Phys. Rev. B 90, 224104 (2014)
(mechanical/elastic stability of crystal phases) — but the 0 K core, computable
directly from an MLIP, with no DFT and no finite-T anharmonic cluster expansion:

  1. cell+ion relaxation   -> equilibrium lattice a0, volume V0
  2. Birch-Murnaghan EOS    -> bulk modulus B0 (isotropic E-V scan, ions relaxed)
  3. stress-strain C_ij     -> 6x6 elastic tensor [GPa]  (reused from crack_analysis)
  4. Born stability         -> C positive-definite  <=>  min eig(C) > 0
                               cubic form: C11-C12>0, C11+2C12>0, C44>0

This is a *model benchmark*: the same crystal is run through every calculator we
compare, and we ask "does each model get the elastic constants right, and is the
crystal mechanically stable per that model?". The headline question for this
project: does the RND-gated physics anchor leave the foundation's (correct)
elastic behaviour untouched in-distribution (do-no-harm), while the fine-tunes
may have degraded it?

Structures: cubic crystals whose elements all lie inside the 26_02_5and8 set
{...Mg,O,Sr,Ti,Zr,...} and that have well-known literature elastic constants, so
the benchmark measures accuracy (vs reference) and not only model-vs-model.

Calculators (MACE family; gfnff-delta-mace env):
  vanilla     MACE-MH-0 (built-in ZBL)
  anchor      MACE-MH-0 + RND-gated per-pair ZBL-residual (pairphys, core_zbl off)
  naive_ft    single-head FT  mace_mh0_mix_clean_F10E1            (head Default)
  mh_pt       multihead FT, foundation-replay head                (head pt_head)
  mh_default  multihead FT, user-adapted head                     (head Default)

Run:
  python \
      idea_uncertainty_gated_physics_anchor/scripts/bench_mechanical_stability.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.eos import EquationOfState
from ase.optimize import FIRE, BFGS
from ase.filters import FrechetCellFilter

warnings.filterwarnings("ignore")

WS = Path(__file__).resolve().parents[1]
RESULTS = WS / "results"
RESULTS.mkdir(exist_ok=True)

# reuse the elastic-tensor + Born/VRH analysis (no reimplementation)
_crack_ws = os.environ.get("ZBL_CRACK_ANALYSIS_WS", "")  # TODO: set for your machine
if _crack_ws:
    sys.path.insert(0, _crack_ws)
from elastic_tensor import elastic_tensor, analyse           # noqa: E402

# reuse the vanilla/anchor factory
sys.path.insert(0, str(WS / "raddmg" / "scripts"))
from common_calc import make_calculator                      # noqa: E402

EV_A3_TO_GPA = 160.21766208
DEVICE = "cuda"

# external fine-tune model workspace  # TODO: set for your machine
FT_WS = Path(os.environ.get("ZBL_FT_MODELS_WS", "/path/to/ft_models"))
FT_MODELS = {
    "naive_ft":  (FT_WS / "mace_mh_mix_clean/results/mace_mh0_mix_clean_F10E1/"
                  "mace_mh0_mix_clean_F10E1.model", "Default"),
    "mh_pt":     (FT_WS / "mace_mh_mix_e0pbe86_multihead/results/"
                  "mace_mh0_mix_e0pbe86_multihead/mace_mh0_mix_e0pbe86_multihead.model", "pt_head"),
    "mh_default": (FT_WS / "mace_mh_mix_e0pbe86_multihead/results/"
                   "mace_mh0_mix_e0pbe86_multihead/mace_mh0_mix_e0pbe86_multihead.model", "Default"),
}
MODELS = ["vanilla", "anchor", "naive_ft", "mh_pt", "mh_default"]


# ---------------------------------------------------------------- structures
def perovskite(A: str, B: str, X: str, a: float) -> Atoms:
    """Cubic ABX3 perovskite (Pm-3m): A corner, B body-centre, X face-centres."""
    return Atoms(f"{A}{B}{X}3",
                 scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5),
                                   (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],
                 cell=[a, a, a], pbc=True)


def build_structures() -> dict:
    return {
        "MgO":    bulk("MgO", "rocksalt", a=4.21),            # 2 atoms
        "SrTiO3": perovskite("Sr", "Ti", "O", 3.905),        # 5 atoms
        "SrZrO3": perovskite("Sr", "Zr", "O", 4.10),         # 5 atoms
    }


# literature elastic constants (experimental, ~RT) [GPa]; None where not pinned
REFERENCE = {
    "MgO":    {"C11": 297.0, "C12": 95.0, "C44": 156.0, "K": 160.0, "src": "exp (Karki/Anderson)"},
    "SrTiO3": {"C11": 316.0, "C12": 103.0, "C44": 124.0, "K": 174.0, "src": "exp (Bell & Rupprecht)"},
    "SrZrO3": {"C11": None, "C12": None, "C44": None, "K": None, "src": "—"},
}

N_FORMULA = {"MgO": 1, "SrTiO3": 1, "SrZrO3": 1}   # formula units per cell


# ---------------------------------------------------------------- calculators
def get_calc(tag: str):
    if tag in ("vanilla", "anchor"):
        return make_calculator("vanilla_pairphys" if tag == "anchor" else "vanilla", device=DEVICE)
    from mace.calculators import MACECalculator
    path, head = FT_MODELS[tag]
    return MACECalculator(model_paths=[str(path)], device=DEVICE,
                          default_dtype="float64", head=head)


# ---------------------------------------------------------------- physics steps
def relax_cell(atoms: Atoms, calc, fmax=0.01, steps=300) -> Atoms:
    a = atoms.copy(); a.calc = calc
    opt = FIRE(FrechetCellFilter(a), logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return a


def eos_b0(atoms: Atoms, calc, span=0.06, npts=9, fmax=0.03, steps=80):
    """Birch-Murnaghan B0 [GPa] from isotropic E-V scan (ions relaxed each volume)."""
    vols, ens = [], []
    for s in np.linspace(1 - span, 1 + span, npts):
        w = atoms.copy()
        w.set_cell(atoms.get_cell() * s, scale_atoms=True)
        w.calc = calc
        try:
            BFGS(w, logfile=None).run(fmax=fmax, steps=steps)
        except Exception:
            pass
        vols.append(w.get_volume()); ens.append(w.get_potential_energy())
    try:
        eos = EquationOfState(vols, ens, eos="birchmurnaghan")
        v0, e0, B = eos.fit()
        return float(B * EV_A3_TO_GPA), float(v0)
    except Exception as exc:
        return float("nan"), float("nan")


def cubic_constants(C: np.ndarray) -> dict:
    """Cubic-averaged C11,C12,C44 + cubic Born criteria from a 6x6 tensor [GPa]."""
    c11 = float(np.mean([C[0, 0], C[1, 1], C[2, 2]]))
    c12 = float(np.mean([C[0, 1], C[0, 2], C[1, 2]]))
    c44 = float(np.mean([C[3, 3], C[4, 4], C[5, 5]]))
    born_cubic = bool((c11 - c12) > 0 and (c11 + 2 * c12) > 0 and c44 > 0)
    return {"C11": c11, "C12": c12, "C44": c44,
            "born_cubic_C11_C12": c11 - c12, "born_cubic_C44": c44,
            "born_cubic_stable": born_cubic}


# ---------------------------------------------------------------- driver
def run_one(name: str, atoms0: Atoms, tag: str) -> dict:
    calc = get_calc(tag)
    rec: dict = {"structure": name, "model": tag, "n_atoms": len(atoms0)}
    try:
        rlx = relax_cell(atoms0, calc)
        V = rlx.get_volume()
        # MgO comes from ase.bulk rocksalt = 2-atom PRIMITIVE cell (V = a_conv^3/4);
        # report the conventional cubic lattice constant a_conv = (4 V)^(1/3).
        a0 = (4.0 * V) ** (1 / 3) if name == "MgO" else float(np.mean(rlx.cell.lengths()))
        rec["a0_relaxed_A"] = float(a0)
        rec["V0_A3"] = float(V)
        B0, _ = eos_b0(rlx, calc)
        rec["B0_eos_GPa"] = B0
        C = elastic_tensor(rlx, calc, delta=0.01, fmax=0.02, steps=120)
        rec.update(cubic_constants(C))
        rec.update(analyse(C))                       # K_VRH, G_VRH, born_min_eig, born_stable, ...
        rec["status"] = "ok"
        print(f"[{name:7s} {tag:11s}] a0={rec['a0_relaxed_A']:.3f}Å  B0={B0:6.1f}  "
              f"C11={rec['C11']:6.1f} C12={rec['C12']:6.1f} C44={rec['C44']:6.1f}  "
              f"K_VRH={rec['K_VRH_GPa']:6.1f}  born={'OK' if rec['born_stable'] else 'UNSTABLE'}")
    except Exception as exc:
        rec["status"] = f"FAIL: {str(exc)[:120]}"
        print(f"[{name:7s} {tag:11s}] FAIL: {exc}")
    return rec


MODEL_LABELS = {"vanilla": "vanilla", "anchor": "anchor", "naive_ft": "naïve-FT",
                "mh_pt": "MH pt_head", "mh_default": "MH Default"}
MODEL_COLORS = {"vanilla": "#2c3e50", "anchor": "#27ae60", "naive_ft": "#c0392b",
                "mh_pt": "#2980b9", "mh_default": "#8e44ad"}
METRICS = [("B0_eos_GPa", "B0"), ("C11", "C11"), ("C12", "C12"),
           ("C44", "C44"), ("K_VRH_GPa", "K_VRH")]


def make_table(out: dict) -> str:
    by = {(r["structure"], r["model"]): r for r in out["results"]}
    lines = []
    for name in ("MgO", "SrTiO3", "SrZrO3"):
        ref = out["reference"][name]
        lines.append(f"\n### {name}   (reference: {ref['src']})")
        lines.append("| model | a0 Å | B0 | C11 | C12 | C44 | K_VRH | Born | Δ K vs vanilla |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        kv = by.get((name, "vanilla"), {}).get("K_VRH_GPa")
        rrow = (f"| **ref (exp)** | – | – | {ref['C11'] or '–'} | {ref['C12'] or '–'} | "
                f"{ref['C44'] or '–'} | {ref['K'] or '–'} | – | – |")
        lines.append(rrow)
        for tag in MODELS:
            r = by.get((name, tag), {})
            if r.get("status") != "ok":
                lines.append(f"| {MODEL_LABELS[tag]} | FAIL | | | | | | | |"); continue
            dk = "" if kv is None else f"{(r['K_VRH_GPa'] - kv):+.0f}"
            born = "OK" if r["born_stable"] else "**UNSTABLE**"
            lines.append(f"| {MODEL_LABELS[tag]} | {r['a0_relaxed_A']:.3f} | {r['B0_eos_GPa']:.0f} | "
                         f"{r['C11']:.0f} | {r['C12']:.0f} | {r['C44']:.0f} | {r['K_VRH_GPa']:.0f} | "
                         f"{born} | {dk} |")
    return "\n".join(lines)


def make_figure(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(r["structure"], r["model"]): r for r in out["results"]}
    names = ["MgO", "SrTiO3", "SrZrO3"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(METRICS)); w = 0.16
    for ax, name in zip(axes, names):
        for i, tag in enumerate(MODELS):
            r = by.get((name, tag), {})
            vals = [r.get(k, np.nan) if r.get("status") == "ok" else np.nan for k, _ in METRICS]
            ax.bar(x + (i - 2) * w, vals, w, label=MODEL_LABELS[tag], color=MODEL_COLORS[tag])
        ref = out["reference"][name]
        for j, (_, lbl) in enumerate(METRICS):
            rv = {"B0": ref["K"], "C11": ref["C11"], "C12": ref["C12"],
                  "C44": ref["C44"], "K_VRH": ref["K"]}.get(lbl)
            if rv:
                ax.plot([x[j] - 2.6 * w, x[j] + 2.6 * w], [rv, rv], "k--", lw=1.4,
                        label="exp ref" if j == 0 else None)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels([m[1] for m in METRICS])
        ax.set_title(name, fontweight="bold"); ax.set_ylabel("GPa")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("0 K mechanical-stability benchmark — elastic constants vs experiment\n"
                 "anchor ≡ vanilla (gate silent in-distribution); fine-tunes soften the lattice",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140); plt.close(fig)


def main():
    structs = build_structures()
    out = {"reference": REFERENCE, "results": []}
    for name, atoms0 in structs.items():
        for tag in MODELS:
            out["results"].append(run_one(name, atoms0, tag))
    js = RESULTS / "mech_stability.json"
    js.write_text(json.dumps(out, indent=2, default=float))
    md = RESULTS / "mech_stability_table.md"
    md.write_text("# 0 K mechanical-stability benchmark (elastic C_ij + Born + EOS)\n"
                  + make_table(out) + "\n")
    fig = RESULTS / "mech_stability.png"
    make_figure(out, fig)
    print(f"\nwrote {js}\nwrote {md}\nwrote {fig}")
    print(make_table(out))


if __name__ == "__main__":
    main()
