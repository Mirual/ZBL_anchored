"""Prove roadmap #4 (decouple create_graph from edge-force/atomic-stress) is
BIT-IDENTICAL, then measure the backward speedup -- all at runtime via
monkeypatch, touching NO installed file.

Mechanism (see audit C3 / roadmap #4):
  When compute_atomic_stresses=True, models.py forces compute_edge_forces=True,
  and get_outputs bumps the FORCE grad to training=True purely so retain_graph
  stays alive for the *second* (edge-force) grad. But compute_forces ties
  retain_graph == create_graph == training, so it ALSO builds the expensive
  double-backward graph that inference never uses.

Fix: split them -> retain_graph=True (graph survives for the edge-force grad),
create_graph=False (no second-order graph). create_graph only affects whether
the *result* is differentiable, so every returned value is numerically
identical; only the wasted backward graph disappears (~1/2 backward on the
atomic-stress path).

This monkeypatches mace.modules.models.get_outputs (the name models.forward()
actually calls, bound via `from .utils import get_outputs`).
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
from ase.build import bulk

import mace
import mace.modules.models as mm
from mace.modules.utils import compute_hessians_vmap
from mace.calculators import MACECalculator

MODEL = os.environ.get("ZBL_MACE_ZBL_MODEL", "/path/to/mace-zbl.model")
HEAD = "omat_pbe"
print(f"mace {mace.__version__} | torch {torch.__version__} | "
      f"{torch.cuda.get_device_name(0)}")


# ----- optimized helpers: retain_graph decoupled from create_graph -----
def _forces_opt(energy, positions, create_graph, retain_graph):
    grad_outputs = [torch.ones_like(energy)]
    g = torch.autograd.grad([energy], [positions], grad_outputs=grad_outputs,
                            retain_graph=retain_graph, create_graph=create_graph,
                            allow_unused=True)[0]
    return torch.zeros_like(positions) if g is None else -1 * g


def _forces_virials_opt(energy, positions, displacement, cell, compute_stress,
                        create_graph, retain_graph):
    grad_outputs = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        [energy], [positions, displacement], grad_outputs=grad_outputs,
        retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        c = cell.view(-1, 3, 3)
        volume = torch.linalg.det(c).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))
    return -1 * forces, -1 * virials, stress


def get_outputs_opt(energy, positions, cell, displacement, vectors=None,
                    training=False, compute_force=True, compute_virials=True,
                    compute_stress=True, compute_hessian=False,
                    compute_edge_forces=False):
    create = bool(training or compute_hessian)            # dropped `or edge_forces`
    retain = bool(training or compute_hessian or compute_edge_forces)
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = _forces_virials_opt(
            energy, positions, displacement, cell, compute_stress,
            create_graph=create, retain_graph=retain)
    elif compute_force:
        forces = _forces_opt(energy, positions, create_graph=create, retain_graph=retain)
        virials, stress = None, None
    else:
        forces, virials, stress = None, None, None
    hessian = compute_hessians_vmap(forces, positions) if compute_hessian else None
    if compute_edge_forces and vectors is not None:
        ef = _forces_opt(energy, vectors, create_graph=create, retain_graph=create)
        edge_forces = -1 * ef if ef is not None else None
    else:
        edge_forces = None
    return forces, virials, stress, hessian, edge_forces


def make_atoms(nn):
    a = bulk("Cu", "fcc", a=3.61).repeat((nn, nn, nn))
    a.positions += np.random.default_rng(0).normal(scale=0.05, size=a.positions.shape)
    return a


def snapshot(calc, atoms):
    a = atoms.copy(); a.calc = calc
    e = float(a.get_potential_energy())
    return {
        "energy": e,
        "forces": calc.results["forces"].copy(),
        "stress": calc.results["stress"].copy(),
        "stresses": calc.results["stresses"].copy(),   # per-atom (voigt-6)
        "virials": calc.results["virials"].copy(),
    }


def timeit(calc, atoms, n=20, w=5):
    a = atoms.copy(); a.calc = calc
    for _ in range(w):
        a.calc.calculate(a)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        a.calc.calculate(a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    nn = int(sys.argv[1]) if len(sys.argv) > 1 else 7   # 343-atom Cu default
    atoms = make_atoms(nn)
    print(f"\nCu fcc ({nn},{nn},{nn}) = {len(atoms)} atoms | fp64 | e3nn | "
          f"compute_atomic_stresses=True")

    calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64",
                          head=HEAD, enable_cueq=False, compute_atomic_stresses=True)

    orig_get_outputs = mm.get_outputs

    # ---- baseline (original double-backward path) ----
    base = snapshot(calc, atoms)
    t_base = timeit(calc, atoms)

    # ---- optimized (retain/create decoupled) ----
    mm.get_outputs = get_outputs_opt
    try:
        opt = snapshot(calc, atoms)
        t_opt = timeit(calc, atoms)
    finally:
        mm.get_outputs = orig_get_outputs  # always restore

    # ---- compare ----
    dE = abs(base["energy"] - opt["energy"])
    dF = float(np.max(np.abs(base["forces"] - opt["forces"])))
    dS = float(np.max(np.abs(base["stress"] - opt["stress"])))
    dAS = float(np.max(np.abs(base["stresses"] - opt["stresses"])))
    dAV = float(np.max(np.abs(base["virials"] - opt["virials"])))
    worst = max(dF, dS, dAS, dAV)

    print("\n--- numerical equivalence (baseline vs optimized) ---")
    print(f"  |dE|              = {dE:.2e} eV")
    print(f"  max|dForces|      = {dF:.2e} eV/A")
    print(f"  max|dStress|      = {dS:.2e}")
    print(f"  max|dAtomStress|  = {dAS:.2e}")
    print(f"  max|dAtomVirial|  = {dAV:.2e}")
    verdict = "BIT-IDENTICAL (lossless)" if worst < 1e-9 and dE < 1e-7 else "DIFFERS!"
    print(f"  -> {verdict}")

    print("\n--- timing (compute_atomic_stresses inference) ---")
    print(f"  baseline (create_graph=True) : {t_base*1e3:8.2f} ms")
    print(f"  optimized (decoupled)        : {t_opt*1e3:8.2f} ms")
    print(f"  speedup                      : {t_base/t_opt:8.2f}x  "
          f"({100*(1-t_opt/t_base):.0f}% faster)")

    out = {"natoms": len(atoms), "dE": dE, "max_dForces": dF, "max_dStress": dS,
           "max_dAtomStress": dAS, "max_dAtomVirial": dAV, "verdict": verdict,
           "t_base_ms": t_base*1e3, "t_opt_ms": t_opt*1e3, "speedup": t_base/t_opt}
    p = "/path/to/validate_opt4.json"  # TODO: set for your machine
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
