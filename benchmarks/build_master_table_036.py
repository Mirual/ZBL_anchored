"""Build the master results tables from the grid_036.out log and print markdown.

Two tables: (A) steady-state throughput ms + speedup vs vanilla (fp64-e3nn at
the SAME size = 1.00x per row), (B) peak GPU memory MB + reduction vs vanilla.
"""
import json

LOG = "/path/to/grid_036.out"  # TODO: set for your machine
COLS = [("e3nn", "float64"), ("e3nn", "float32"), ("cueq", "float64"),
        ("cueq", "float32"), ("cmp-e3nn", "float32"), ("cmp+cueq", "float32")]
HEAD = ["fp64 e3nn (vanilla)", "fp32 e3nn", "fp64 cueq", "fp32 cueq",
        "fp32 cmp-e3nn", "fp32 cmp+cueq"]

rows = [json.loads(l[len("RESULT_JSON: "):]) for l in open(LOG)
        if l.startswith("RESULT_JSON: ")]
data = {}  # (natoms) -> {(combo,dtype): row}
for r in rows:
    if not r.get("ok"):
        continue
    data.setdefault(r["natoms"], {})[(r["combo"], r["dtype"])] = r
sizes = sorted(data)


def cell(d, key, field, ref):
    r = d.get(key)
    if r is None or r.get(field) is None:
        return "—"
    v = r[field]
    if ref is None:
        return f"{v:.1f}"
    return f"{v:.1f} ({ref/v:.2f}×)"


def table(field, caption):
    print(f"\n**{caption}**\n")
    print("| atoms | " + " | ".join(HEAD) + " |")
    print("|" + "---|" * (len(HEAD) + 1))
    for n in sizes:
        d = data[n]
        van = d.get(("e3nn", "float64"))
        ref = van[field] if van and van.get(field) is not None else None
        cells = [cell(d, k, field, ref) for k in COLS]
        print(f"| {n} | " + " | ".join(cells) + " |")


table("steady_ms", "A. Throughput — steady-state of a single point (ms) and ×vs vanilla (fp64-e3nn) of the same size")
table("peak_mem_MB", "B. Peak GPU memory (MB) and ×smaller vs vanilla of the same size")

# headline maxima across the whole grid
best_sp = max(((data[n][("e3nn","float64")]["steady_ms"]/data[n][k]["steady_ms"], n, k)
               for n in sizes for k in COLS
               if ("e3nn","float64") in data[n] and k in data[n]
               and data[n][k].get("steady_ms")), default=(0,0,0))
best_mem = max(((data[n][("e3nn","float64")]["peak_mem_MB"]/data[n][k]["peak_mem_MB"], n, k)
                for n in sizes for k in COLS
                if ("e3nn","float64") in data[n] and k in data[n]
                and data[n][k].get("peak_mem_MB")), default=(0,0,0))
print(f"\n**Grid maxima:** speedup **{best_sp[0]:.2f}×** "
      f"({best_sp[2][1]} {best_sp[2][0]}, {best_sp[1]} atoms); "
      f"memory **{best_mem[0]:.2f}×** smaller "
      f"({best_mem[2][1]} {best_mem[2][0]}, {best_mem[1]} atoms).")
