#!/usr/bin/env python3
"""Summary markdown table from runs/bench_*.json."""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"

rows = []
for f in sorted(RUNS.glob("bench_*.json")):
    d = json.loads(f.read_text())
    tag = f.stem.replace("bench_", "")
    rows.append((tag, d))

print("| run | calculator | atoms | mode | ms/step | of which calculator | drift, meV/atom |")
print("|---|---|---|---|---|---|---|")
for tag, d in rows:
    if "ms_per_step" not in d:
        continue
    drift = d.get("etotal_drift_meV_per_atom")
    drift_s = f"{drift:+.3f}" if d["mode"] == "nve" else "—"
    print(f"| {tag} | {d['calc']} | {d['natoms']} | {d['mode'].upper()} "
          f"| {d['ms_per_step']:.1f} | {d['ms_calc_per_step']:.1f} | {drift_s} |")
