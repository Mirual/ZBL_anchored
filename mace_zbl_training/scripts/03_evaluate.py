#!/usr/bin/env python3
"""Compute metrics + parity plots for MACE-MH-{0,1} zero-shot predictions.

Reads results/predictions_mh{0,1}.json, fits per-element shift (LSQ on
E_ref - E_pred against composition counts — see innp/CLAUDE.md), reports
raw and shift-corrected MAE/RMSE/R² in eV/structure and meV/atom, writes
results/metrics.json and results/parity_{tag}.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WS = Path(__file__).resolve().parents[1]
RESULTS = WS / "results"

sys.path.insert(0, str(WS.parent / "innp" / "finetune"))
from importlib import import_module

_metrics = import_module("06_compare_metrics")
mae = _metrics.mae
rmse = _metrics.rmse
r2 = _metrics.r2
fit_element_shift = _metrics.fit_element_shift
build_count_matrix = _metrics.build_count_matrix

Z_TO_SYM = {
    1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na",
    12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K",
    20: "Ca", 22: "Ti", 25: "Mn", 26: "Fe", 38: "Sr", 39: "Y", 40: "Zr",
    43: "Tc", 53: "I", 55: "Cs", 90: "Th", 92: "U", 93: "Np",
}


def metric_block(E_ref: np.ndarray, E_pred: np.ndarray, n_atoms: np.ndarray) -> dict:
    de = E_ref - E_pred
    de_per_atom = de / n_atoms
    return {
        "mae_eV":            mae(E_ref, E_pred),
        "rmse_eV":           rmse(E_ref, E_pred),
        "r2":                r2(E_ref, E_pred),
        "mae_meV_per_atom":  float(np.mean(np.abs(de_per_atom)) * 1000.0),
        "rmse_meV_per_atom": float(np.sqrt(np.mean(de_per_atom ** 2)) * 1000.0),
        "mean_signed_eV":    float(np.mean(de)),
    }


def parity_plot(E_ref, E_pred, title, png_path, rmse_eV, mae_eV):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(E_ref, E_pred, s=12, alpha=0.7)
    lo = float(min(E_ref.min(), E_pred.min()))
    hi = float(max(E_ref.max(), E_pred.max()))
    pad = 0.02 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("E_ref (DFT) [eV]")
    ax.set_ylabel("E_pred [eV]")
    ax.set_title(f"{title}\nRMSE={rmse_eV:.3f} eV  MAE={mae_eV:.3f} eV")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def evaluate_one(path: Path) -> dict:
    with open(path) as f:
        payload = json.load(f)
    tag = payload.get("tag") or payload["model"]
    frames = payload["frames"]

    E_ref = np.array([fr["E_ref"] for fr in frames], dtype=float)
    E_pred = np.array([fr["E_pred"] for fr in frames], dtype=float)
    n_atoms = np.array([fr["n_atoms"] for fr in frames], dtype=float)
    counts, elems = build_count_matrix(frames)

    raw = metric_block(E_ref, E_pred, n_atoms)
    c_vec, E_shifted = fit_element_shift(E_ref, E_pred, counts)
    shifted = metric_block(E_ref, E_shifted, n_atoms)

    shifts_per_element = {
        Z_TO_SYM.get(int(z), f"Z{int(z)}"): float(c_vec[j])
        for j, z in enumerate(elems)
    }

    parity_plot(
        E_ref, E_shifted,
        title=f"MACE-{tag.upper()} (+ZBL) zero-shot, per-element shift",
        png_path=RESULTS / f"parity_{tag}.png",
        rmse_eV=shifted["rmse_eV"], mae_eV=shifted["mae_eV"],
    )

    return {
        "model": tag,
        "n_frames": len(frames),
        "raw": raw,
        "shift_corrected": shifted,
        "shift_per_element_eV": shifts_per_element,
    }


def main() -> None:
    out = {}
    paths = sorted(RESULTS.glob("predictions_*.json"))
    if not paths:
        sys.exit("no results/predictions_*.json found — run 02_predict.py first")
    for path in paths:
        with open(path) as f:
            tag = json.load(f).get("tag") or path.stem.replace("predictions_", "")
        print(f"evaluating {tag}  ({path.name})")
        out[tag] = evaluate_one(path)
        m = out[tag]
        print(f"  raw      MAE={m['raw']['mae_eV']:.3f} eV  "
              f"({m['raw']['mae_meV_per_atom']:.1f} meV/atom)  R²={m['raw']['r2']:.4f}")
        print(f"  shifted  MAE={m['shift_corrected']['mae_eV']:.3f} eV  "
              f"({m['shift_corrected']['mae_meV_per_atom']:.1f} meV/atom)  "
              f"R²={m['shift_corrected']['r2']:.4f}")

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RESULTS/'metrics.json'}")


if __name__ == "__main__":
    main()
