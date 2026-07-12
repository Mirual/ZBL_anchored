#!/usr/bin/env python3
"""Port-glue for DPA-3.1: single entry point for the calculator + per-atom descriptors.

DPA analogue of MACE `calc.get_descriptors(at)` — a DIRECT call into the DPA descriptor submodule (dim=128),
without the buggy `eval_descriptor` (see PROBLEMS.md §1–5). All other anchor layers (pair_physics,
RND gate, SelectiveNet calibration) are model-agnostic. Run in the deepmd_env conda environment.
"""
from __future__ import annotations
import os
import numpy as np

# Fixed inference batch (in atoms) — for the E/F path (predict/tune `get_potential_energy`/
# `get_forces`), so that DeePMD auto-batch probing on a free 98 GB GPU does not pick too large a
# starting batch (false OOM). No longer affects descriptor extraction — it bypasses auto-batch.
os.environ.setdefault("DP_INFER_BATCH_SIZE", "2048")

# --- vanilla DPA-3.1-3M-FT foundation (no built-in ZBL — the anchor carries all short-range) ---
FOUNDATION = os.environ.get("ZBL_DPA_MODEL", "/path/to/dpa-3.1-3m-ft.pth")

# data (shared with MACE-anchor for direct comparability)
SPLITS = os.environ.get("ZBL_EVAL_DATA", "/path/to/vasp_eval/preflight")
RND_CORPUS = os.environ.get("ZBL_MIXED_DATA", "/path/to/mixed_dataset/data") + "/mixed_train.xyz"   # MPtrj-like background for RND
MPTRJ = os.environ.get("ZBL_MPTRJ_XYZ", "/path/to/mptrj_stratified_10k.xyz")

DESC_DIM = 128


def load_dp(model: str = FOUNDATION):
    from deepmd.calculator import DP
    return DP(model=model)


# --- per-atom descriptors: DIRECT submodule call (clean path, resolves PROBLEMS.md §1–5) ---
# Previously went through `eval_descriptor` (hook + scripted accumulator + force autograd) — 5 problems:
# non-resettable list (#1), GPU leak ~0.5 GB/frame (#2), false auto-batch OOM (#3),
# JIT recompilation on every fresh DP (#4), ~0.75 s/frame on UNneeded force computation (#5).
# The direct `atomic_model.descriptor(...)` under no_grad avoids all five: one persistent DP, JIT once,
# no hook/accumulator/forces/leak. ~22 ms/frame (×34), flat GPU memory. Descriptors match the old
# method bit-for-bit (max|Δ|<1e-6, verified).

_DESC_CTX: dict = {}


def _desc_ctx(model: str) -> dict:
    """Per-process cache: persistent DP + descriptor submodule + its rcut/sel/device/dtype."""
    if model not in _DESC_CTX:
        calc = load_dp(model)
        top = calc.dp.deep_eval.dp.model["Default"]
        desc = top.atomic_model.descriptor
        p = next(desc.parameters())
        _DESC_CTX[model] = dict(calc=calc, tm=calc.dp.get_type_map(), desc=desc,
                                rcut=float(top.get_rcut()), sel=top.get_sel(), dev=p.device, dt=p.dtype)
    return _DESC_CTX[model]


def compute_descriptors(frames, model: str = FOUNDATION):
    """Per-atom DPA descriptors [nat,128] per frame → list in input order (DPA analogue of MACE get_descriptors).

    Direct call into the descriptor submodule under no_grad (without the `eval_descriptor` bugs, see block above).
    The nlist is built by `extend_input_and_build_neighbor_list`: ghost shifts are computed in float64 (otherwise
    einsum fails Double-vs-Float), while the descriptor itself uses the model dtype (float32).
    """
    import torch
    from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
    c = _desc_ctx(model)
    tm, desc, rcut, sel, dev, dt = c["tm"], c["desc"], c["rcut"], c["sel"], c["dev"], c["dt"]
    out = []
    for at in frames:
        pos = torch.tensor(at.get_positions(), dtype=torch.float64, device=dev).reshape(1, -1, 3)
        atype = torch.tensor([[tm.index(s) for s in at.get_chemical_symbols()]], dtype=torch.long, device=dev)
        box = torch.tensor(np.asarray(at.cell).reshape(1, 9), dtype=torch.float64, device=dev) if at.pbc.any() else None
        with torch.no_grad():
            ec, ea, mp, nl = extend_input_and_build_neighbor_list(pos, atype, rcut, sel, mixed_types=True, box=box)
            d = desc(ec.to(dt), ea, nl, mapping=mp)[0]
        out.append(d[0].detach().cpu().numpy())
    return out
