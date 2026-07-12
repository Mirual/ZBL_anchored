#!/usr/bin/env python3
"""Merge the per-calculator Tier-0 JSON outputs into one summary + comparison tables.

    python aggregate.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("ZBL_ANCHOR_WS", "/path/to/idea_uncertainty_gated_physics_anchor"))  # TODO: set for your machine
RESULTS_DIR = ROOT / "raddmg" / "results"


def load(pattern):
    return [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob(pattern))]


def main() -> None:
    summary = {"eos": {}, "defect": {}, "recoil": {}}

    # --- T0.1 EOS: B0 per (host, calc) ---
    print("\n=== T0.1 EOS — bulk modulus B0 [GPa] ===")
    for d in load("t0_eos_*.json"):
        calc = d["calc"]
        for host, r in d["hosts"].items():
            summary["eos"].setdefault(host, {})[calc] = r.get("B0_GPa")
    for host, by in summary["eos"].items():
        cols = "  ".join(f"{c}={(v if v is None else round(v,1))}" for c, v in by.items())
        print(f"  {host:5s}  {cols}")

    # --- T0.2 defects: dE per (host, calc, defect) ---
    print("\n=== T0.2 defect formation — dE [eV] (mu-free unless noted) ===")
    for d in load("t0_defect_*.json"):
        host, calc = d["host"], d["calc"]
        summary["defect"].setdefault(host, {})[calc] = {
            k: dict(dE=v["dE"], mu_free=v["mu_free"]) for k, v in d["defects"].items()}
    for host, by in summary["defect"].items():
        print(f"  [{host}]")
        names = sorted({n for c in by.values() for n in c})
        for n in names:
            cells = "  ".join(
                f"{c}={by[c][n]['dE']:+.3f}" for c in by if n in by[c])
            mu = "" if all(by[c][n]["mu_free"] for c in by if n in by[c]) else " (μ-dep)"
            print(f"    {n:10s} {cells}{mu}")

    # --- T0.3 recoil: peak n_frenkel per (host, calc) + onset energy ---
    print("\n=== T0.3 recoil — surviving Frenkel pairs vs energy ===")
    for d in load("t0_recoil_*.json"):
        host, calc = d["host"], d["calc"]
        rows = [r for r in d["rows"] if "n_frenkel" in r]
        onset = min((r["energy_eV"] for r in rows if r["n_frenkel"] > 0), default=None)
        peak = max((r["n_frenkel"] for r in rows), default=0)
        ncrash = sum(1 for r in rows if r.get("crash"))
        summary["recoil"].setdefault(host, {})[calc] = dict(
            onset_eV=onset, peak_frenkel=peak, n_cases=len(rows), n_crash=ncrash)
        print(f"  {host:5s} {calc:16s} onset={onset} eV  peak_FP={peak}  "
              f"cases={len(rows)}  crashes={ncrash}")

    out = RESULTS_DIR / "RADDMG_SUMMARY.json"
    out.write_text(json.dumps(summary, indent=1))
    print("\n→", out)


if __name__ == "__main__":
    main()
