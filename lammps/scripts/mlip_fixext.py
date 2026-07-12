#!/usr/bin/env python3
"""Universal MLIP launch in LAMMPS: MACE / DPA-3.1 / anchor variants.

Mechanism: python-driven LAMMPS + `fix external pf/callback` — at each step
LAMMPS provides coordinates, the calculator (GPU) returns E/F/virial. One MPI-rank
(one GPU) → the full cell as a whole, bit-for-bit the same path as the ASE evals.

Run (deepmd env + pylibs):
  PYTHONPATH=<...>/lammps/pylibs python mlip_fixext.py \
      --calc mace --data data.lammps --elems Cu --mode nvt --steps 1000 --temp 600

Calculators:
  mace         MACE-MH-1 (head omat_pbe), fp32 + cuEquivariance — fast production mode
  mace-anchor  MACE-MH-0 + RND-gate + per-pair ZBL-residual (AnchorCalculator as is)
  dpa          DPA-3.1 (frozen .pth) via DeepPot — without refreezing/border_op
  dpa-anchor   DPA-3.1 + RND-gate + pairphys (θ from pairphys_theta.json)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ANCHOR_WS = Path(os.environ.get("ZBL_ANCHOR_WS", ""))  # external anchor workspace (MACE)
DPA_WS = Path(os.environ.get("ZBL_DPA_WS", ""))  # external anchor workspace (DPA-3.1)
MACE_MH1 = os.environ.get("ZBL_MACE_MH1", "/path/to/mace-mh-1.model")
DPA_MODEL = os.environ.get("ZBL_DPA_MODEL", "/path/to/dpa-3.1-3m-ft.pth")
MACE_DIMER_PKL = ANCHOR_WS / "results" / "dimer_tables" / "dimer_zblON_user_wbm.pkl"

# ASE voigt: (xx,yy,zz,yz,xz,xy) -> LAMMPS virial: (xx,yy,zz,xy,xz,yz)
VOIGT_TO_LAMMPS = [0, 1, 2, 5, 4, 3]


def build_atoms(data_file: str, elems: list[str], lmp) -> "Atoms":
    """ASE Atoms from LAMMPS data (types -> elements), positions updated in callback."""
    from ase import Atoms

    natoms = lmp.get_natoms()
    types = np.asarray(lmp.gather_atoms("type", 0, 1))  # tag order
    numbers = np.zeros(natoms, dtype=int)
    from ase.data import atomic_numbers

    for t, sym in enumerate(elems, start=1):
        numbers[types == t] = atomic_numbers[sym]
    if (numbers == 0).any():
        raise SystemExit("--elems does not cover all atom types from the data file")
    return Atoms(numbers=numbers, positions=np.zeros((natoms, 3)), pbc=True)


def cell_from_lmp(lmp) -> np.ndarray:
    (xlo, ylo, zlo), (xhi, yhi, zhi), xy, yz, xz, _, _ = lmp.extract_box()
    return np.array(
        [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]]
    )


class MaceBackend:
    """MACE via ASE calculator (fp32 + cueq) or AnchorCalculator."""

    def __init__(self, args):
        self.with_stress = getattr(args, "with_stress", True)
        if args.calc == "mace":
            from mace.calculators import MACECalculator

            kwargs = dict(
                model_paths=[args.model or MACE_MH1],
                device="cuda",
                default_dtype=args.dtype,
                head=args.head,
                enable_cueq=not args.no_cueq,
            )
            if getattr(args, "compile", False):
                # compile+cueq: works with mace>=0.3.16 (in 0.3.15 compile
                # auto-disables with a warning)
                kwargs["compile_mode"] = "default"
            self.calc = MACECalculator(**kwargs)
        else:  # mace-anchor
            sys.path.insert(0, str(ANCHOR_WS / "md_stability" / "scripts"))
            sys.path.insert(0, str(ANCHOR_WS / "scripts"))
            import anchor_calculator as ac_mod
            from anchor_calculator import AnchorCalculator

            lite = getattr(args, "anchor_lite", False)
            if getattr(args, "anchor_cueq", False):
                # cueq engine inside anchor: swap the symbol during
                # construction so the hook/DimerCache live on the
                # cueq model right away (without a second copy of weights on GPU)
                orig_cls = ac_mod.MACECalculator
                ac_mod.MACECalculator = (
                    lambda **kw: orig_cls(enable_cueq=True, **kw))
                try:
                    self.calc = AnchorCalculator(
                        mode=args.anchor_mode, device="cuda",
                        core_zbl=args.core_zbl,
                        dimer_cache_path=(None if lite else str(MACE_DIMER_PKL)
                                          if MACE_DIMER_PKL.exists() else None),
                    )
                finally:
                    ac_mod.MACECalculator = orig_cls
            else:
                self.calc = AnchorCalculator(
                    mode=args.anchor_mode, device="cuda",
                    core_zbl=args.core_zbl,
                    dimer_cache_path=(None if lite else str(MACE_DIMER_PKL)
                                      if MACE_DIMER_PKL.exists() else None),
                )
            # speed up the corr part (profile @864: 20 ms of which 10 — slow
            # ase.neighborlist; when the gate is silent the correction is identically 0):
            # 1) matscipy neighbour_list (same API, an order of magnitude faster);
            # 2) early exit if the gate is silent for all atoms.
            # ANCHOR_NO_FASTCORR=1 — restore the original path (for A/B).
            if os.environ.get("ANCHOR_NO_FASTCORR") != "1":
                import anchor_calculator as _ac
                import rnd_pairphys_predict as _rpp
                from anchor_predict import smoothstep as _smoothstep
                try:
                    from matscipy.neighbours import neighbour_list as _msnl
                    _rpp.neighbor_list = _msnl
                except ImportError:
                    pass
                _orig_corr = _ac.pairphys_corr

                def _fast_corr(at, nov, dc, r_lo, r_hi, lam, power, **kw):
                    if _smoothstep(nov, r_lo, r_hi).max() <= 1e-6:
                        return 0.0, np.zeros((len(at), 3))
                    return _orig_corr(at, nov, dc, r_lo, r_hi, lam, power, **kw)

                _ac.pairphys_corr = _fast_corr

            # do not hold the autograd graph between steps: their hook saves
            # node_feats with grad_fn; ours (fires next) detaches it.
            # ANCHOR_NO_DETACH=1 — only for A/B memory measurements.
            if os.environ.get("ANCHOR_NO_DETACH") != "1":
                calc_ref = self.calc

                def _detach_nf(_mod, _inp, out):
                    if calc_ref._nf is not None:
                        calc_ref._nf = calc_ref._nf.detach()

                self.calc.mace.models[0].register_forward_hook(_detach_nf)
            if lite:
                # lightweight pack: same gate + dimer tables in packed-fp32
                lite_dir = Path(__file__).resolve().parent.parent / "anchor_lite"
                from make_anchor_lite import unpack
                from rnd_anchor_predict import RNDGate

                self.calc.dc._cache = unpack(lite_dir / "dimer_packed.pkl")
                self.calc.gate = RNDGate("cuda")  # weights are the same (copy in the pack)

    def evaluate(self, atoms):
        atoms.calc = self.calc
        e = float(atoms.get_potential_energy())
        f = np.asarray(atoms.get_forces())
        v6 = None
        if self.with_stress:
            try:
                s = np.asarray(atoms.get_stress())  # voigt, eV/A^3
                vol = atoms.get_volume()
                v6 = (-vol * s)[VOIGT_TO_LAMMPS]
            except Exception:
                pass
        return e, f, v6


class DpaBackend:
    """DPA-3.1 via DeepPot (+ optionally RND-gate + pairphys correction)."""

    def __init__(self, args):
        from deepmd.infer.deep_pot import DeepPot

        self.dp = DeepPot(args.model or DPA_MODEL)
        self.tmap = self.dp.get_type_map()
        self.anchor = args.calc == "dpa-anchor"
        self.with_stress = getattr(args, "with_stress", True)
        if self.anchor:
            sys.path.insert(0, str(DPA_WS / "scripts"))
            import torch
            from gate import RNDGate, smoothstep
            from pair_physics import DimerCache
            import predict as _pred
            from predict import corr
            import dpa_common

            # speed up corr: matscipy neighbours + early exit when the gate is silent
            if os.environ.get("ANCHOR_NO_FASTCORR") != "1":
                try:
                    from matscipy.neighbours import neighbour_list as _msnl
                    _pred.neighbor_list = _msnl
                except ImportError:
                    pass
                _orig = corr

                def corr(at, nov, dc, r_lo, r_hi, lam, power, **kw):
                    if smoothstep(nov, r_lo, r_hi).max() <= 1e-6:
                        return 0.0, np.zeros((len(at), 3))
                    return _orig(at, nov, dc, r_lo, r_hi, lam, power, **kw)

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.gate = RNDGate(dev)
            self.dc = DimerCache(
                dpa_common.load_dp(args.model or DPA_MODEL),
                cache_path=str(DPA_WS / "results" / "dimer_dpa.pkl"),
            )
            self.corr = corr
            self.compute_descriptors = dpa_common.compute_descriptors
            th = json.loads((DPA_WS / "results" / "pairphys_theta.json").read_text())
            self.theta = (th["r_lo"], th["r_hi"], th["lam"], th["power"])
            # fused-eval: E/F and gate descriptors in ONE network pass
            # (descriptor -> fitting_net -> out_bias -> autograd). Removes
            # the second descriptor forward (~2/3 of the anchor step cost).
            # Validated: cu256 dF<=2.7e-7; keep51 — within the model's own
            # run-to-run noise (dynamic-sel atomicAdd). Virial is not computed
            # in fused -> on stress request auto-fallback to two passes.
            # DPA_NO_FUSED=1 — force the old path.
            self.fused_ctx = None
            if os.environ.get("DPA_NO_FUSED") != "1":
                ctx = dict(dpa_common._desc_ctx(args.model or DPA_MODEL))
                ctx["am"] = (ctx["calc"].dp.deep_eval.dp
                             .model["Default"].atomic_model)
                self.fused_ctx = ctx
        self._atype = None

    def _fused_eval(self, atoms):
        """(E, F, desc) in a single forward+backward via scripted submodules."""
        import torch
        from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

        c = self.fused_ctx
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64,
                           device=c["dev"], requires_grad=True).reshape(1, -1, 3)
        atype = torch.tensor([self._atype], dtype=torch.long, device=c["dev"])
        box = (torch.tensor(np.asarray(atoms.cell).reshape(1, 9),
                            dtype=torch.float64, device=c["dev"])
               if atoms.pbc.any() else None)
        ec, ea, mp, nl = extend_input_and_build_neighbor_list(
            pos, atype, c["rcut"], c["sel"], mixed_types=True, box=box)
        d = c["desc"](ec.to(c["dt"]), ea, nl, mapping=mp)[0]
        e_at = c["am"].fitting_net(d, atype)["energy"].double()
        e_at = e_at + c["am"].out_bias.to(c["dev"])[0][atype[0]].unsqueeze(0)
        energy = e_at.sum()
        (grad,) = torch.autograd.grad(energy, pos)
        return (float(energy), -grad[0].detach().cpu().numpy(),
                d[0].detach().cpu().numpy())

    def evaluate(self, atoms):
        if self._atype is None:
            self._atype = [self.tmap.index(s) for s in atoms.get_chemical_symbols()]
        with_stress = getattr(self, "with_stress", True)
        if self.anchor and self.fused_ctx is not None and not with_stress:
            # single pass: E/F + gate descriptors (virial not needed — NVT/NVE)
            e, f, desc = self._fused_eval(atoms)
            v6 = None
        else:
            e, f, v = self.dp.eval(
                atoms.positions.reshape(1, -1),
                np.asarray(atoms.cell).reshape(1, -1),
                self._atype,
            )
            e = float(np.asarray(e).ravel()[0])
            f = np.asarray(f).reshape(-1, 3)
            v9 = np.asarray(v).reshape(3, 3)
            v6 = np.array([v9[0, 0], v9[1, 1], v9[2, 2],
                           v9[0, 1], v9[0, 2], v9[1, 2]])
            desc = None
        if self.anchor:
            if desc is None:
                desc = self.compute_descriptors([atoms])[0]
            nov = self.gate.novelty(desc)
            r_lo, r_hi, lam, power = self.theta
            ec, fc = self.corr(atoms, nov, self.dc, r_lo, r_hi, lam, power)
            e += ec
            f = f + fc
            # do not add correction virial (contribution ≈0 when the gate is silent; NVT/NVE don't depend on it)
        return e, f, v6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calc", required=True,
                    choices=["mace", "mace-anchor", "dpa", "dpa-anchor"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--elems", nargs="+", required=True,
                    help="elements in LAMMPS type order: --elems Cu O ...")
    ap.add_argument("--mode", default="validate",
                    choices=["validate", "nve", "nvt", "custom"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warm", type=int, default=5,
                    help="warmup steps before timing (for --compile need 30+)")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--timestep", type=float, default=0.001)
    ap.add_argument("--cmds", help="file with LAMMPS commands for --mode custom")
    ap.add_argument("--out", default=None, help="json output (validate/bench metrics)")
    ap.add_argument("--log", default="log.mlip")
    # mace
    ap.add_argument("--model", default=None)
    ap.add_argument("--head", default="omat_pbe")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--no-cueq", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile (mace>=0.3.16, env mace-cueq-test)")
    # anchor
    ap.add_argument("--anchor-mode", default="pairphys",
                    choices=["pairphys", "bornmayer"])
    ap.add_argument("--core-zbl", action="store_true")
    ap.add_argument("--anchor-lite", action="store_true",
                    help="lightweight anchor pack (anchor_lite/): packed-fp32 dimer, "
                         "3.9 MB addition instead of 6.8; numerically equivalent")
    ap.add_argument("--anchor-cueq", action="store_true",
                    help="cuEquivariance inside anchor-MACE (less GPU memory "
                         "and faster); requires novelty validation on your system")
    ap.add_argument("--stress", default="auto", choices=["auto", "on", "off"],
                    help="virial in LAMMPS: auto = on for validate/custom (NPT), "
                         "off for nve/nvt (saves the second forward for MACE)")
    args = ap.parse_args()
    args.with_stress = (args.stress == "on") or (
        args.stress == "auto" and args.mode in ("validate", "custom"))

    # build the backend BEFORE liblammps: python libs (h5py/torch) must load
    # their native dependencies first, otherwise conflict with libhdf5 from the LAMMPS env
    backend = (MaceBackend if args.calc.startswith("mace") else DpaBackend)(args)

    from lammps import lammps

    lmp = lammps(cmdargs=["-log", args.log, "-screen", "none"])
    lmp.commands_string(
        f"""
units metal
boundary p p p
atom_style atomic
read_data {args.data}
fix mlip all external pf/callback 1 1
fix_modify mlip energy yes virial yes
"""
    )
    atoms = build_atoms(args.data, args.elems, lmp)
    state = {"ncalls": 0, "last_e": None, "last_f": None, "t_calc": 0.0}

    def callback(caller, ntimestep, nlocal, tag, x, f):
        t0 = time.perf_counter()
        order = np.argsort(tag)
        atoms.set_cell(cell_from_lmp(lmp), scale_atoms=False)
        atoms.positions[:] = x[order]
        e, forces, v6 = backend.evaluate(atoms)
        f[order] = forces
        lmp.fix_external_set_energy_global("mlip", e)
        if v6 is not None:
            lmp.fix_external_set_virial_global("mlip", list(v6))
        state["ncalls"] += 1
        state["last_e"] = e
        state["last_f"] = forces.copy()
        state["t_calc"] += time.perf_counter() - t0

    lmp.set_fix_external_callback("mlip", callback, lmp)

    result = {"calc": args.calc, "mode": args.mode, "natoms": lmp.get_natoms()}
    if args.mode == "validate":
        lmp.command("run 0 post no")
        result.update(
            energy_eV=state["last_e"],
            forces=state["last_f"].tolist(),
            pe_thermo=lmp.get_thermo("pe"),
            press_bar=lmp.get_thermo("press"),
        )
        print(f"E = {state['last_e']:.8f} eV; pe(thermo) = {result['pe_thermo']:.8f}; "
              f"P = {result['press_bar']:.1f} bar")
    elif args.mode in ("nve", "nvt"):
        integ = ("fix md all nve" if args.mode == "nve"
                 else f"fix md all nvt temp {args.temp} {args.temp} 0.1")
        lmp.commands_string(
            f"""
velocity all create {args.temp} 4928459 mom yes rot yes dist gaussian
{integ}
timestep {args.timestep}
thermo 100
"""
        )
        lmp.command(f"run {args.warm} post no")   # warmup (JIT/compilation/caches)
        e0 = lmp.get_thermo("etotal")
        state["t_calc"] = 0.0
        t0 = time.perf_counter()
        lmp.command(f"run {args.steps} post no")
        wall = time.perf_counter() - t0
        e1 = lmp.get_thermo("etotal")
        result.update(
            steps=args.steps,
            ms_per_step=wall * 1000 / args.steps,
            ms_calc_per_step=state["t_calc"] * 1000 / args.steps,
            etotal_drift_meV_per_atom=(e1 - e0) / lmp.get_natoms() * 1000,
            final_T=lmp.get_thermo("temp"),
            final_pe=lmp.get_thermo("pe"),
        )
        print(f"{args.mode.upper()} {args.steps} steps: {result['ms_per_step']:.2f} ms/step "
              f"(calculator {result['ms_calc_per_step']:.2f}); "
              f"drift {result['etotal_drift_meV_per_atom']:+.3f} meV/atom; "
              f"T={result['final_T']:.0f}K")
    else:  # custom
        if not args.cmds:
            raise SystemExit("--mode custom requires --cmds file")
        lmp.commands_string(Path(args.cmds).read_text())
        result.update(final_pe=lmp.get_thermo("pe"), ncalls=state["ncalls"])

    if args.out:
        Path(args.out).write_text(json.dumps(result))
    lmp.close()


if __name__ == "__main__":
    main()
