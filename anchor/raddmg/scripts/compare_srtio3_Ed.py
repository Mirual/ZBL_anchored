#!/usr/bin/env python3
"""Compare SrTiO3 recoil onset (≈ threshold displacement energy E_d) to literature.

Onset per (species, direction) = lowest recoil energy that leaves ≥1 surviving Frenkel
pair (from t0_recoil_SrTiO3_*.json). Reports per-species directional-min and direction-
mean E_d for vanilla vs vanilla_pairphys, next to DFT-MD / experimental references, so
we can finally judge whether the anchor's force change moves E_d toward the truth.

    python compare_srtio3_Ed.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
RES = ROOT / "raddmg" / "results"
FIG = ROOT / "raddmg" / "figures"

# SrTiO3 reference E_d [eV]: DFT-MD weighted-avg / directional-min (Uberuaga, ORNL),
# plus a classical-MD / experimental band for context.
LIT = {
    "O":  dict(dft_avg=35.7, dft_min=13, other="cls 40–50; exp(CaTiO3) 45±4"),
    "Sr": dict(dft_avg=53.5, dft_min=25, other="cls 67–80"),
    "Ti": dict(dft_avg=64.9, dft_min=38, other="cls 70–140; exp(CaTiO3-Ti) 69±9"),
}


def onsets(calc: str) -> dict:
    f = RES / f"t0_recoil_SrTiO3_{calc}.json"
    if not f.exists():
        return {}
    rows = [r for r in json.loads(f.read_text())["rows"] if "n_frenkel" in r]
    by = {}
    for r in rows:
        by.setdefault((r["species"], r["direction"]), []).append(
            (r["energy_eV"], r["n_frenkel"]))
    out = {}
    for (sp, dr), lst in by.items():
        lst.sort()
        on = next((e for e, fp in lst if fp >= 1), None)  # threshold along this direction
        out.setdefault(sp, {})[dr] = on
    return out


def main() -> None:
    o_van, o_pp = onsets("vanilla"), onsets("vanilla_pairphys")
    res = {}
    print(f"{'sp':>3} | {'E_d^min van':>11} {'E_d^min pp':>10} | "
          f"{'E_d⟨dir⟩ van':>12} {'E_d⟨dir⟩ pp':>11} | {'DFT min/avg':>12} | other")
    for sp in ("O", "Sr", "Ti"):
        row = {"literature": LIT[sp]}
        for calc, o in (("vanilla", o_van), ("vanilla_pairphys", o_pp)):
            vals = [v for v in o.get(sp, {}).values() if v is not None]
            row[calc] = dict(per_dir=o.get(sp, {}),
                             Ed_min=(min(vals) if vals else None),
                             Ed_mean=(round(float(np.mean(vals)), 1) if vals else None))
        res[sp] = row
        vmn, pmn = row["vanilla"]["Ed_min"], row["vanilla_pairphys"]["Ed_min"]
        vme, pme = row["vanilla"]["Ed_mean"], row["vanilla_pairphys"]["Ed_mean"]
        print(f"{sp:>3} | {str(vmn):>11} {str(pmn):>10} | {str(vme):>12} {str(pme):>11} | "
              f"{LIT[sp]['dft_min']}/{LIT[sp]['dft_avg']:>5} | {LIT[sp]['other']}")

    (RES / "srtio3_Ed_compare.json").write_text(json.dumps(res, indent=1))
    print("→", RES / "srtio3_Ed_compare.json")

    # figure: directional-min E_d, vanilla vs pairphys vs DFT(min, avg)
    sps = ["O", "Sr", "Ti"]
    x = np.arange(len(sps))
    w = 0.2
    g = lambda key, sub: [res[s][key].get(sub) if isinstance(res[s][key], dict) else None
                          for s in sps]
    van = [res[s]["vanilla"]["Ed_min"] for s in sps]
    pp = [res[s]["vanilla_pairphys"]["Ed_min"] for s in sps]
    dmin = [LIT[s]["dft_min"] for s in sps]
    davg = [LIT[s]["dft_avg"] for s in sps]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 1.5 * w, [v or 0 for v in van], w, label="vanilla E_d^min")
    ax.bar(x - 0.5 * w, [v or 0 for v in pp], w, label="pairphys E_d^min")
    ax.bar(x + 0.5 * w, dmin, w, label="DFT-MD min (lit)")
    ax.bar(x + 1.5 * w, davg, w, label="DFT-MD avg (lit)")
    ax.set_xticks(x)
    ax.set_xticklabels(sps)
    ax.set_ylabel("threshold displacement E_d [eV]")
    ax.set_title("SrTiO₃ — E_d: MLIP recoil onset vs literature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "srtio3_Ed_compare.png", dpi=140)
    plt.close(fig)
    print("→", FIG / "srtio3_Ed_compare.png")


if __name__ == "__main__":
    main()
