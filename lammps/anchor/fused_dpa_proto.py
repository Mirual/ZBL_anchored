#!/usr/bin/env python3
"""Prototype of fused DPA eval: E/F + descriptors in ONE network pass.

Idea: the DeepPot E/F forward and the gate's descriptor pass evaluate the same
network twice. Here E/F is assembled by hand from the scripted submodules
(descriptor -> fitting_net -> out_bias -> autograd), and the descriptor falls
out for free from the same pass.

Validation: E/F against dp.eval, descriptors against dpa_common.compute_descriptors,
on cu256 (normal) and keep51 (extreme). Timing: fused vs two passes.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ws = os.environ.get("ZBL_DPA_ANCHOR_WS", "")
if _ws:
    sys.path.insert(0, _ws)
import dpa_common
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

MODEL = dpa_common.FOUNDATION


def fused_eval(ctx, atoms):
    """(E, F, desc) in a single forward+backward."""
    tm, desc, rcut, sel, dev, dt = (ctx["tm"], ctx["desc"], ctx["rcut"],
                                    ctx["sel"], ctx["dev"], ctx["dt"])
    am = ctx["am"]
    nloc = len(atoms)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=dev,
                       requires_grad=True).reshape(1, -1, 3)
    atype = torch.tensor([[tm.index(s) for s in atoms.get_chemical_symbols()]],
                         dtype=torch.long, device=dev)
    box = (torch.tensor(np.asarray(atoms.cell).reshape(1, 9),
                        dtype=torch.float64, device=dev)
           if atoms.pbc.any() else None)
    ec, ea, mp, nl = extend_input_and_build_neighbor_list(
        pos, atype, rcut, sel, mixed_types=True, box=box)
    out = desc(ec.to(dt), ea, nl, mapping=mp)
    d = out[0]                                    # [1, nloc, 128]
    e_at = am.fitting_net(d, atype)["energy"]     # [1, nloc, 1]
    bias = am.out_bias.to(dev)[0]                 # [ntypes, 1] fp64
    e_at = e_at.double() + bias[atype[0]].unsqueeze(0)
    energy = e_at.sum()
    (force,) = torch.autograd.grad(energy, pos)
    return (float(energy), -force[0].detach().cpu().numpy(),
            d[0].detach().cpu().numpy())


def main():
    from ase.data import atomic_numbers
    from ase.io import read

    ctx = dict(dpa_common._desc_ctx(MODEL))
    ctx["am"] = ctx["calc"].dp.deep_eval.dp.model["Default"].atomic_model
    dp_calc = ctx["calc"]

    cases = []
    z = np.load("../dpa_smoke/cu256.npz")
    from ase import Atoms
    cases.append(("cu256", Atoms("Cu256", positions=z["coord"], cell=z["cell"], pbc=True)))
    zmap = {i + 1: atomic_numbers[s] for i, s in enumerate(["Mn", "O", "S", "Sr"])}
    k51 = read("data_keep51.lammps", format="lammps-data", Z_of_type=zmap, style="atomic")
    k51.pbc = True
    cases.append(("keep51", k51))

    for tag, at in cases:
        # reference: dp.eval + a separate descriptor pass
        probe = at.copy()
        probe.calc = dp_calc
        e_ref = float(probe.get_potential_energy())
        f_ref = np.asarray(probe.get_forces())
        d_ref = dpa_common.compute_descriptors([at])[0]
        e, f, d = fused_eval(ctx, at)
        print(f"{tag}: dE = {e - e_ref:.3e} eV | max|dF| = {np.abs(f - f_ref).max():.3e} "
              f"(Fmax={np.abs(f_ref).max():.1f}) | max|d_desc| = {np.abs(d - d_ref).max():.3e}")

    # timing on cu256
    at = cases[0][1]
    probe = at.copy(); probe.calc = dp_calc
    rng = np.random.default_rng(0)

    def two_pass():
        probe.positions += rng.normal(0, 1e-4, probe.positions.shape)
        probe.get_forces()
        dpa_common.compute_descriptors([probe])

    def one_pass():
        at.positions += rng.normal(0, 1e-4, at.positions.shape)
        fused_eval(ctx, at)

    for fn in (two_pass, one_pass):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        print(f"{fn.__name__}: {(time.perf_counter() - t0) * 100:.1f} ms/step")


if __name__ == "__main__":
    main()
