#!/usr/bin/env python3
"""Relative percentage-error distributions for the 2x2 model-vs-dataset grid.

Inputs (results/):
    predictions_mh0_user.json   - vanilla MACE-MH-0  on 26_02_5and8 test
    predictions_ft_user.json    - FT MACE-MH-0       on 26_02_5and8 test
    predictions_mh0_mptrj.json  - vanilla MACE-MH-0  on MPtrj stratified-100
    predictions_ft_mptrj.json   - FT MACE-MH-0       on MPtrj stratified-100

Metric definitions (per user, 2026-05-12):
    Energy:  eps_E[k]    = |E_pred[k] - E_ref[k]| / |E_ref[k]| * 100        (per structure)
    Forces:  eps_F[k,i]  = ||F_pred[k,i] - F_ref[k,i]|| / ||F_ref[k,i]|| * 100  (per atom)
             - no floor; drop atoms with ||F_ref[k,i]|| == 0 (division by zero).
             - report n_atoms_zero_fref.

Outputs (results/):
    distribution_compare.json
    distribution_compare_energy.png
    distribution_compare_forces.png
    distribution_compare_summary.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WS = Path(__file__).resolve().parents[1]
RESULTS = WS / "results"

CONDITIONS: list[tuple[str, str, str]] = [
    # (key, label, predictions filename)
    ("mh0_user",   "vanilla x user (26_02_5and8)",  "predictions_mh0_user.json"),
    ("ft_user",    "fine-tuned x user (26_02_5and8)", "predictions_ft_user.json"),
    ("mh0_mptrj",  "vanilla x MPtrj-100",            "predictions_mh0_mptrj.json"),
    ("ft_mptrj",   "fine-tuned x MPtrj-100",         "predictions_ft_mptrj.json"),
]

COLORS: dict[str, str] = {
    "mh0_user":  "#1f77b4",
    "ft_user":   "#2ca02c",
    "mh0_mptrj": "#ff7f0e",
    "ft_mptrj":  "#d62728",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dist-compare")


@dataclass(frozen=True)
class ConditionStats:
    key: str
    label: str
    n_frames_total: int
    n_frames_used: int
    n_frames_dropped_oov: int
    n_atoms_total: int
    n_atoms_used: int
    n_atoms_dropped_zero_fref: int
    eps_E: np.ndarray   # shape (n_frames_used,)
    eps_F: np.ndarray   # shape (n_atoms_used,)

    def summary(self) -> dict:
        def stats(a: np.ndarray) -> dict:
            if a.size == 0:
                return {"n": 0}
            return {
                "n": int(a.size),
                "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)),
                "p99": float(np.percentile(a, 99)),
                "max": float(np.max(a)),
            }

        return {
            "key": self.key,
            "label": self.label,
            "n_frames_total": self.n_frames_total,
            "n_frames_used": self.n_frames_used,
            "n_frames_dropped_oov": self.n_frames_dropped_oov,
            "n_atoms_total": self.n_atoms_total,
            "n_atoms_used": self.n_atoms_used,
            "n_atoms_dropped_zero_fref": self.n_atoms_dropped_zero_fref,
            "energy_pct": stats(self.eps_E),
            "force_pct": stats(self.eps_F),
        }


def load_condition(key: str, label: str, filename: str) -> ConditionStats:
    path = RESULTS / filename
    payload = json.loads(path.read_text())
    frames = payload["frames"]

    eps_e_vals: list[float] = []
    eps_f_vals: list[float] = []
    n_frames_dropped = 0
    n_atoms_total = 0
    n_atoms_zero = 0

    for fr in frames:
        n_atoms_total += int(fr["n_atoms"])
        if fr.get("E_pred") is None or fr.get("F_pred") is None:
            n_frames_dropped += 1
            continue

        e_ref = float(fr["E_ref"])
        e_pred = float(fr["E_pred"])
        if e_ref == 0.0:
            # would divide by zero; treat as dropped frame for energy
            n_frames_dropped += 1
            continue
        eps_e_vals.append(abs(e_pred - e_ref) / abs(e_ref) * 100.0)

        if fr.get("F_ref") is None:
            # no reference forces; cannot compute per-atom percentages
            continue
        f_ref = np.asarray(fr["F_ref"], dtype=float)   # (n_atoms, 3)
        f_pred = np.asarray(fr["F_pred"], dtype=float)
        fref_mag = np.linalg.norm(f_ref, axis=1)         # (n_atoms,)
        fdiff_mag = np.linalg.norm(f_pred - f_ref, axis=1)
        mask = fref_mag > 0.0
        n_atoms_zero += int((~mask).sum())
        ratios = fdiff_mag[mask] / fref_mag[mask] * 100.0
        eps_f_vals.extend(ratios.tolist())

    n_frames_used = len(eps_e_vals)
    return ConditionStats(
        key=key,
        label=label,
        n_frames_total=len(frames),
        n_frames_used=n_frames_used,
        n_frames_dropped_oov=n_frames_dropped,
        n_atoms_total=n_atoms_total,
        n_atoms_used=len(eps_f_vals),
        n_atoms_dropped_zero_fref=n_atoms_zero,
        eps_E=np.asarray(eps_e_vals, dtype=float),
        eps_F=np.asarray(eps_f_vals, dtype=float),
    )


def plot_distributions(
    conditions: list[ConditionStats],
    attr: str,
    title: str,
    xlabel: str,
    out_path: Path,
    p_clip: float = 99.5,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # determine common log-spaced bins across all non-empty conditions
    pool: list[np.ndarray] = []
    for c in conditions:
        a = getattr(c, attr)
        if a.size:
            pool.append(a[a > 0])  # log scale ignores 0; tiny share, harmless
    if not pool:
        log.warning("no data for %s; writing empty figure", attr)
    else:
        all_vals = np.concatenate(pool)
        hi = float(np.percentile(all_vals, p_clip))
        lo = float(max(np.min(all_vals), 1e-6))
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)

        for c in conditions:
            a = getattr(c, attr)
            if not a.size:
                continue
            ax.hist(
                np.clip(a, lo, hi),
                bins=bins,
                histtype="step",
                linewidth=2,
                density=True,
                label=f"{c.label}  (n={a.size}, med={np.median(a):.2f}%)",
                color=COLORS[c.key],
            )
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("wrote %s", out_path)


def write_summary_md(conditions: list[ConditionStats], path: Path) -> None:
    rows = ["| condition | n_frames | n_atoms | E median % | E p90 % | F median % | F p90 % | drop_oov | drop_F=0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in conditions:
        s = c.summary()
        e = s["energy_pct"]
        f = s["force_pct"]
        rows.append(
            f"| {c.label} | {s['n_frames_used']} | {s['n_atoms_used']} | "
            f"{e.get('median', float('nan')):.2f} | {e.get('p90', float('nan')):.2f} | "
            f"{f.get('median', float('nan')):.2f} | {f.get('p90', float('nan')):.2f} | "
            f"{s['n_frames_dropped_oov']} | {s['n_atoms_dropped_zero_fref']} |"
        )
    body = "\n".join(rows)
    path.write_text(
        "# Relative percentage-error distributions\n\n"
        "Energy: per structure  eps_E = |E_pred - E_ref| / |E_ref| * 100\n\n"
        "Forces: per atom       eps_F = ||F_pred - F_ref|| / ||F_ref|| * 100  "
        "(atoms with ||F_ref|| = 0 dropped, no floor)\n\n"
        + body + "\n"
    )
    log.info("wrote %s", path)


def main() -> None:
    conditions = [load_condition(k, label, fn) for (k, label, fn) in CONDITIONS]

    summary = {c.key: c.summary() for c in conditions}
    (RESULTS / "distribution_compare.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", RESULTS / "distribution_compare.json")

    plot_distributions(
        conditions, "eps_E",
        title="Relative energy error  |ΔE|/|E_ref|",
        xlabel="energy error (%)",
        out_path=RESULTS / "distribution_compare_energy.png",
    )
    plot_distributions(
        conditions, "eps_F",
        title="Relative force error  ||ΔF||/||F_ref||  (per atom)",
        xlabel="force error (%)",
        out_path=RESULTS / "distribution_compare_forces.png",
    )
    write_summary_md(conditions, RESULTS / "distribution_compare_summary.md")


if __name__ == "__main__":
    main()
