#!/usr/bin/env python3
"""Build a lightweight anchor pack: same computation, minimal addition to vanilla.

Weight facts:
  - mace-mh-0.model is already fp32 (39.9 MB) — model shared with vanilla, not in the pack.
  - rnd.pt (gate) is already fp32, 1.32 MB — taken as is.
  - dimer cache 5.48 MB — of which useful data is ~0.6 MB: 3570 pairs x
    (3 arrays x 15 float64) drowned in pickle overhead (~820 B/pair).
    We pack into stacked float32 arrays in one block.

Accuracy: interpolating dV/dVdr in fp32 changes forces by ~1e-7 relative —
far below fp32 inference noise. Speed: np.interp is the same.

Output: lammps/anchor_lite/{rnd.pt, dimer_packed.pkl}
"""
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

SRC_RND = os.path.join(os.environ.get("ZBL_ANCHOR_RESULTS", "results"), "rnd.pt")
SRC_DIMER = os.path.join(os.environ.get("ZBL_ANCHOR_RESULTS", "results"),
                         "dimer_tables", "dimer_zblON_user_wbm.pkl")
OUT = Path(__file__).resolve().parents[1] / "anchor_lite"
OUT.mkdir(exist_ok=True)


def mb(p):
    return Path(p).stat().st_size / 1e6


cache = pickle.loads(Path(SRC_DIMER).read_bytes())
keys = sorted(cache.keys())
n = len(keys)
lens = np.asarray([len(cache[k]["r"]) for k in keys], dtype=np.int32)

packed = {
    "format": "dimer-packed-v1",                        # ragged concatenation
    "keys": np.asarray(keys, dtype=np.int16),           # [n, 2] (Zi, Zj)
    "lens": lens,                                       # grid length of each pair
    "r": np.concatenate([cache[k]["r"] for k in keys]).astype(np.float32),
    "dV": np.concatenate([cache[k]["dV"] for k in keys]).astype(np.float32),
    "dVdr": np.concatenate([cache[k]["dVdr"] for k in keys]).astype(np.float32),
    "r_cut": np.asarray([cache[k]["r_cut"] for k in keys], dtype=np.float32),
}
(OUT / "dimer_packed.pkl").write_bytes(pickle.dumps(packed, protocol=4))
print(f"dimer: {mb(SRC_DIMER):.2f} MB -> {mb(OUT / 'dimer_packed.pkl'):.2f} MB "
      f"({n} pairs, grids {lens.min()}–{lens.max()} points)")

shutil.copy2(SRC_RND, OUT / "rnd.pt")
print(f"rnd:   {mb(OUT / 'rnd.pt'):.2f} MB (fp32, as is)")

tot = mb(OUT / "dimer_packed.pkl") + mb(OUT / "rnd.pt")
print(f"\nanchor addition: 6.80 MB -> {tot:.2f} MB "
      f"(+{tot/39.9*100:.1f}% to the mh-0 model 39.9 MB)")


def unpack(path):
    """Back to the DimerCache._cache format (used by the driver)."""
    p = pickle.loads(Path(path).read_bytes())
    assert p.get("format") == "dimer-packed-v1"
    off = np.concatenate([[0], np.cumsum(p["lens"])])
    out = {}
    for i, (zi, zj) in enumerate(p["keys"]):
        s = slice(off[i], off[i + 1])
        out[(int(zi), int(zj))] = {
            "r": p["r"][s], "dV": p["dV"][s], "dVdr": p["dVdr"][s],
            "r_cut": float(p["r_cut"][i]),
        }
    return out


if __name__ == "__main__":
    # round-trip self-check
    back = unpack(OUT / "dimer_packed.pkl")
    rel = 0.0
    for k in keys:
        for f in ("r", "dV", "dVdr"):
            a, b = np.asarray(cache[k][f]), np.asarray(back[k][f])
            d = np.abs(b - a) / np.maximum(np.abs(a), 1e-12)
            rel = max(rel, float(d.max()))
    print(f"round-trip max relative error (fp64->fp32): {rel:.2e}")
