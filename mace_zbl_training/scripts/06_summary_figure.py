#!/usr/bin/env python3
"""Single composite figure summarizing the relative %-error distribution comparison.

Layout:
    row 1 (full width): stats table (one row per condition)
    row 2 left : energy %-error histograms (log x)
    row 2 right: force  %-error histograms (log x)

Reads results/distribution_compare.json and the four prediction JSONs;
writes results/distribution_compare_overview.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WS = Path(__file__).resolve().parents[1]
RESULTS = WS / "results"

CONDITIONS: list[tuple[str, str, str]] = [
    ("mh0_user",   "vanilla x user (26_02_5and8)",    "predictions_mh0_user.json"),
    ("ft_user",    "fine-tuned x user (26_02_5and8)", "predictions_ft_user.json"),
    ("mh0_mptrj",  "vanilla x MPtrj-100",             "predictions_mh0_mptrj.json"),
    ("ft_mptrj",   "fine-tuned x MPtrj-100",          "predictions_ft_mptrj.json"),
]
COLORS = {"mh0_user": "#1f77b4", "ft_user": "#2ca02c",
          "mh0_mptrj": "#ff7f0e", "ft_mptrj": "#d62728"}


def load_eps(key: str, fname: str) -> tuple[np.ndarray, np.ndarray, int]:
    payload = json.loads((RESULTS / fname).read_text())
    eE: list[float] = []
    eF: list[float] = []
    dropped = 0
    for fr in payload["frames"]:
        if fr.get("E_pred") is None or fr.get("F_pred") is None:
            dropped += 1
            continue
        e_ref = float(fr["E_ref"])
        if e_ref == 0.0:
            dropped += 1
            continue
        eE.append(abs(float(fr["E_pred"]) - e_ref) / abs(e_ref) * 100.0)
        if fr.get("F_ref") is None:
            continue
        f_ref = np.asarray(fr["F_ref"], dtype=float)
        f_pred = np.asarray(fr["F_pred"], dtype=float)
        mag_ref = np.linalg.norm(f_ref, axis=1)
        mag_diff = np.linalg.norm(f_pred - f_ref, axis=1)
        mask = mag_ref > 0.0
        eF.extend((mag_diff[mask] / mag_ref[mask] * 100.0).tolist())
    return np.asarray(eE), np.asarray(eF), dropped


def hist_panel(ax, arrays, labels, colors, title, xlabel, p_clip=99.5) -> None:
    pool = np.concatenate([a[a > 0] for a in arrays if a.size]) if arrays else np.array([])
    if pool.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return
    hi = float(np.percentile(pool, p_clip))
    lo = float(max(np.min(pool), 1e-6))
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    for a, lab, col in zip(arrays, labels, colors):
        if not a.size:
            continue
        ax.hist(
            np.clip(a, lo, hi), bins=bins,
            histtype="step", linewidth=2, density=True,
            label=f"{lab}\n  n={a.size}, med={np.median(a):.2f}%", color=col,
        )
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main() -> None:
    summary = json.loads((RESULTS / "distribution_compare.json").read_text())
    data = {key: (label, *load_eps(key, fname)) for key, label, fname in CONDITIONS}

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.4], hspace=0.30, wspace=0.20)

    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.axis("off")
    headers = ["condition", "n_frames",
               "E med %", "E p90 %", "E p99 %",
               "F med %", "F p90 %", "F p99 %",
               "drop_oov", "drop_F=0"]
    rows = []
    for key, _label, _fname in CONDITIONS:
        s = summary[key]
        e = s["energy_pct"]; f = s["force_pct"]
        rows.append([
            s["label"],
            f"{s['n_frames_used']}/{s['n_frames_total']}",
            f"{e.get('median', float('nan')):.2f}",
            f"{e.get('p90', float('nan')):.2f}",
            f"{e.get('p99', float('nan')):.2f}",
            f"{f.get('median', float('nan')):.2f}",
            f"{f.get('p90', float('nan')):.2f}",
            f"{f.get('p99', float('nan')):.2f}",
            str(s["n_frames_dropped_oov"]),
            str(s["n_atoms_dropped_zero_fref"]),
        ])
    tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    for i, (key, _, _) in enumerate(CONDITIONS, start=1):
        for j in range(len(headers)):
            tbl[(i, j)].set_facecolor(COLORS[key])
            tbl[(i, j)].set_alpha(0.18)
    ax_tbl.set_title(
        "Relative %-error summary  —  eps_E = |E_pred - E_ref| / |E_ref| * 100  (per structure);  "
        "eps_F = ||F_pred - F_ref|| / ||F_ref|| * 100  (per atom, no floor)",
        fontsize=11, pad=14,
    )

    labels = [data[k][0] for k, _, _ in CONDITIONS]
    eE_arrays = [data[k][1] for k, _, _ in CONDITIONS]
    eF_arrays = [data[k][2] for k, _, _ in CONDITIONS]
    colors = [COLORS[k] for k, _, _ in CONDITIONS]

    ax_e = fig.add_subplot(gs[1, 0])
    hist_panel(ax_e, eE_arrays, labels, colors,
               title="Energy error  |ΔE|/|E_ref|", xlabel="energy error (%)")
    ax_f = fig.add_subplot(gs[1, 1])
    hist_panel(ax_f, eF_arrays, labels, colors,
               title="Force error  ||ΔF||/||F_ref||  (per atom)", xlabel="force error (%)")

    fig.suptitle(
        "MACE-MH-0  —  fine-tuned vs vanilla  —  user 26_02_5and8 vs authors' MPtrj-100",
        fontsize=13, y=0.995,
    )
    out = RESULTS / "distribution_compare_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
