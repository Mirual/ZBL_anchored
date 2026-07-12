#!/usr/bin/env python3
"""Compare two bench_mace.py JSON runs: accuracy parity + speedup.

Typical use — prove the patched build is bit-identical and time it:
    python compare_runs.py stock.json ours.json

Same dtype/config -> energy/forces/stress should match to ~round-off (PASS).
If you compare different configs (e.g. fp64 vs fp32, or +cueq), the diffs show
the accuracy cost of that speed knob; use --tol to set the pass threshold.
"""

import argparse
import json

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def max_abs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ref", help="baseline JSON (e.g. stock.json)")
    p.add_argument("test", help="comparison JSON (e.g. ours.json)")
    p.add_argument("--etol", type=float, default=1e-6, help="max |dE| per atom, eV")
    p.add_argument("--ftol", type=float, default=1e-6, help="max |dF|, eV/A")
    p.add_argument("--stol", type=float, default=1e-7, help="max |d stress|, eV/A^3")
    args = p.parse_args()

    ref, test = load(args.ref), load(args.test)
    print(f"REF : {ref['label']:<10} {ref['config']}")
    print(f"TEST: {test['label']:<10} {test['config']}\n")

    by_label = {r["label"]: r for r in test["results"]}
    worst_e = worst_f = worst_s = 0.0
    print(f"{'structure':>12} {'dE/atom[eV]':>13} {'dFmax[eV/A]':>13} "
          f"{'dStress':>11} {'t_ref[ms]':>10} {'t_test[ms]':>11} {'speedup':>8}")
    for r in ref["results"]:
        t = by_label.get(r["label"])
        if t is None:
            print(f"{r['label']:>12}   (missing in test run)")
            continue
        de = abs(r["energy_per_atom"] - t["energy_per_atom"])
        df = max_abs(r["forces"], t["forces"])
        ds = max_abs(r["stress"], t["stress"])
        worst_e, worst_f, worst_s = max(worst_e, de), max(worst_f, df), max(worst_s, ds)
        spd = r["sp_time_s"] / t["sp_time_s"] if t["sp_time_s"] else float("nan")
        print(f"{r['label']:>12} {de:13.2e} {df:13.2e} {ds:11.2e} "
              f"{r['sp_time_s']*1e3:10.1f} {t['sp_time_s']*1e3:11.1f} {spd:7.2f}x")

    total_spd = ref["total_sp_time_s"] / test["total_sp_time_s"]
    print("\n" + "-" * 72)
    print(f"worst |dE/atom| = {worst_e:.2e} eV   (tol {args.etol:.0e})")
    print(f"worst |dF|      = {worst_f:.2e} eV/A (tol {args.ftol:.0e})")
    print(f"worst |dStress| = {worst_s:.2e} eV/A^3 (tol {args.stol:.0e})")
    print(f"overall speedup = {total_spd:.2f}x  "
          f"({ref['total_sp_time_s']*1e3:.1f} -> {test['total_sp_time_s']*1e3:.1f} ms)")

    ok = worst_e <= args.etol and worst_f <= args.ftol and worst_s <= args.stol
    print("\nACCURACY: " + ("PASS (within tolerance — outputs match)" if ok
                            else "DIFF  (above tolerance — expected if configs differ)"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
