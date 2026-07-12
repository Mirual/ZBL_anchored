"""Orchestrate the mace 0.3.16 validation as isolated worker subprocesses.

Each combo runs in its own process (validate_036_worker.py) so host RAM and GPU
memory are fully reclaimed between combos -> no OOM, and torch.compile state is
never shared. Aggregates three things:

  [1] lossless: 0.3.16 e3nn fp64 energies vs recorded 0.3.15 (mh-1 numerics)
  [2] engage:   does compile+cueq actually engage in 0.3.16 (the #6 unblock)?
  [3] bench:    fp32 timing across sizes -> does compile add on top of cueq?

Writes runs/validate_036.json and prints human tables.
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "validate_036_worker.py")
PY = sys.executable
OUT_JSON = os.path.join(HERE, "runs", "validate_036.json")

COMBOS = ["e3nn", "cueq", "cmp-e3nn", "cmp+cueq"]


def run_worker(task, combo="e3nn", dtype="float64", sizes="3,5,7,9"):
    cmd = [PY, WORKER, "--task", task, "--combo", combo, "--dtype", dtype, "--sizes", sizes]
    sys.stderr.write(f"\n>>> {' '.join(cmd[1:])}\n")
    p = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON: ")), None)
    if line is None:
        sys.stderr.write(f"!!! no RESULT_JSON (rc={p.returncode})\n"
                         f"--- stderr tail ---\n{p.stderr[-1500:]}\n")
        return {"task": task, "combo": combo, "ok": False, "rc": p.returncode,
                "stderr_tail": p.stderr[-800:]}
    return json.loads(line[len("RESULT_JSON: "):])


def main():
    report = {}

    # ---------- [1] lossless ----------
    print("=== [1] lossless mh-1 load: 0.3.16 e3nn fp64 vs recorded 0.3.15 ===")
    r1 = run_worker("lossless")
    report["lossless"] = r1
    worst = 0.0
    if r1.get("results"):
        print(f"{'label':>16}{'E 0.3.16[eV]':>18}{'E 0.3.15[eV]':>18}{'dE/atom[eV]':>14}")
        for x in r1["results"]:
            d = abs(x["e16"] - x["e15"]) / x["natoms"]
            worst = max(worst, d)
            print(f"{x['label']:>16}{x['e16']:>18.6f}{x['e15']:>18.6f}{d:>14.2e}")
        verdict = "LOSSLESS" if worst < 1e-5 else "DIFFERS (model numerics changed!)"
        print(f"worst |dE/atom| (0.3.16 vs 0.3.15) = {worst:.2e} eV  ->  {verdict}")
        report["lossless_worst_dE_per_atom"] = worst
        report["lossless_verdict"] = verdict
    else:
        print("  FAILED:", r1)

    # ---------- [2] does compile+cueq engage? ----------
    print("\n=== [2] combos on 0.3.16 (KEY: does compile+cueq engage?) ===")
    engage = {}
    e_ref = None
    for combo in COMBOS:
        r = run_worker("engage", combo=combo)
        engage[combo] = r
        if not r.get("ok"):
            print(f"  {combo:>10}: BUILD/RUN FAILED: {r.get('error', r.get('stderr_tail',''))[:80]}")
            continue
        if combo == "e3nn":
            e_ref = r["energy"]
        de = abs(r["energy"] - e_ref) / r["natoms"] if e_ref is not None else float("nan")
        warn = " | WARN:compile-disabled" if r.get("warn_compile_disabled") else ""
        print(f"  {combo:>10}: ok | use_compile={r.get('use_compile')} | dE/at={de:.2e}{warn}")
    report["engage"] = engage

    # interpret #6 unblock: compile+cueq engages iff use_compile True AND no disable warning
    cc = engage.get("cmp+cueq", {})
    unblocked = bool(cc.get("ok") and cc.get("use_compile") is True
                     and not cc.get("warn_compile_disabled"))
    report["compile_cueq_unblocked_036"] = unblocked
    print(f"\n  -> compile+cueq engaged in 0.3.16? {'YES (#6 unblocked)' if unblocked else 'NO (still gated)'}")

    # ---------- [3] benchmark fp32 across sizes ----------
    print("\n=== [3] benchmark fp32 across sizes (does compile add on top of cueq?) ===")
    bench = {}
    for combo in COMBOS:
        bench[combo] = run_worker("bench", combo=combo, dtype="float32")
    report["bench"] = bench

    sizes = sorted({int(s) for c in bench.values() for s in (c.get("times_ms") or {})})
    print(f"{'natoms':>7}" + "".join(f"{c:>13}" for c in COMBOS))
    for s in sizes:
        row = f"{s:>7}"
        t_e3nn = (bench.get("e3nn", {}).get("times_ms") or {}).get(str(s))
        for combo in COMBOS:
            t = (bench.get(combo, {}).get("times_ms") or {}).get(str(s))
            if t is None:
                row += f"{'-':>13}"
            else:
                spd = (t_e3nn / t) if t_e3nn else float("nan")
                row += f"{t:7.0f}/{spd:4.2f}x"
        print(row)
    print("\n(format: t[ms]/speedup-vs-e3nn)")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(report, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
