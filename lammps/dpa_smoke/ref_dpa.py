#!/usr/bin/env python3
"""Reference E/F for cu256 via the python DeepPot (deepmd env, GPU)."""
import json
import os

import numpy as np
from deepmd.infer.deep_pot import DeepPot

MODEL = os.environ.get("ZBL_DPA_MODEL", "/path/to/dpa-3.1-3m-ft.pth")

z = np.load("cu256.npz")
dp = DeepPot(MODEL)
tm = dp.get_type_map()
cu_idx = tm.index("Cu")
atype = np.full(len(z["atype"]), cu_idx, dtype=int)

e, f, v = dp.eval(
    z["coord"].reshape(1, -1), z["cell"].reshape(1, -1), atype.tolist()
)
out = {
    "energy_eV": float(e[0][0]),
    "forces": np.asarray(f[0]).reshape(-1, 3).tolist(),
    "model": MODEL,
    "type_map_index_Cu": cu_idx,
}
with open("ref_dpa.json", "w") as fh:
    json.dump(out, fh)
print(f"E = {out['energy_eV']:.6f} eV; Fmax = {np.abs(f).max():.4f} eV/A")
