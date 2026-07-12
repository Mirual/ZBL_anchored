#!/usr/bin/env python3
"""Parse MACE training log(s) and plot loss / RMSE_E / RMSE_F vs epoch.

Reads any log files passed on the CLI (or default e-prio + f-prio runs in
mace_mh_zbl/logs). Saves results/training_curves.png.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path(__file__).resolve().parents[1]

LINE_RE = re.compile(
    r"Epoch (\d+): head: \w+, loss=([\d.]+), "
    r"RMSE_E_per_atom=([\d.]+) meV, RMSE_F=([\d.]+) meV / A"
)


def parse(log_path: Path):
    epochs, loss, rmse_e, rmse_f = [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            rmse_e.append(float(m.group(3)) / 1000.0)   # meV → eV
            rmse_f.append(float(m.group(4)) / 1000.0)
    return epochs, loss, rmse_e, rmse_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=None,
                    help="log files (label=path); default: e-prio + f-prio")
    ap.add_argument("--out", default=str(WS / "results" / "training_curves.png"))
    args = ap.parse_args()

    if args.logs is None:
        runs = [
            ("e-prio (10:1)", WS / "logs" / "finetune_mh0.log"),
            ("f-prio (1:10) [killed]", WS / "logs" / "finetune_mh0_fprio.log"),
        ]
    else:
        runs = []
        for spec in args.logs:
            if "=" in spec:
                label, p = spec.split("=", 1)
            else:
                label, p = Path(spec).stem, spec
            runs.append((label, Path(p)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for (label, path), color in zip(runs, colors):
        if not path.exists():
            print(f"  skip {label}: {path} missing")
            continue
        ep, loss, e, f = parse(path)
        if not ep:
            print(f"  skip {label}: no epoch lines parsed")
            continue
        print(f"  {label:30s}  {len(ep):4d} epochs  best RMSE_E={min(e):6.2f} eV/atom @ epoch {ep[e.index(min(e))]}")
        axes[0].plot(ep, loss, color=color, label=label, lw=1.4)
        axes[1].plot(ep, e,    color=color, label=label, lw=1.4)
        axes[2].plot(ep, f,    color=color, label=label, lw=1.4)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation loss"); axes[0].set_title("loss (log scale)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("RMSE_E [eV/atom]"); axes[1].set_title("Energy RMSE per atom")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("RMSE_F [eV/Å]");   axes[2].set_title("Force RMSE per component")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"MACE-MH-0 (+ZBL) fine-tune on 26_02_5and8 — validation curves   "
        f"[zero-shot baseline: 42.3 eV/atom, R²=0.978]",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
