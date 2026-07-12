# anchor/ — the uncertainty-gated physical anchor (MACE core)

This is the heart of the project: the physics anchor applied on top of the
MACE-MH-0 foundation model. See the top-level `README.md` for the method and
results. Every external path is read from an environment variable — copy the
repo-root `config.example.env` to `config.env`, fill it in, and `source` it.

## `scripts/` — method, calibration, evaluation

**Shared primitives (imported by the predict scripts — keep them together):**
- `anchor_predict.py` — baseline output-additive anchor `E = E_MACE + V_BM(1-smoothstep)`;
  provides `smoothstep`/`dsmoothstep`/`pair_correction` reused across the family.
- `pair_physics.py` — `DimerCache` (per-pair ZBL-minus-own-dimer residual), `zbl_V`, `zbl_grad`.
- `rho_anchor_predict.py` — kNN/PCA extrapolation gate `Rho` + the gated kernel `pair_corr_gated`.
- `rnd_anchor_predict.py` — the **RND novelty gate** (`RNDGate`) that replaces the fragile kNN gate.
- `rnd_pairphys_predict.py` — the flagship: RND gate + per-pair physics residual (+ optional
  divergent-ZBL core for the keV–MeV radiation regime).
- `dimer_residual_predict.py` — per-pair dimer-residual predictor.

**Build the artifacts first** (these produce files under `$ZBL_ANCHOR_RESULTS`, gitignored):
1. `rnd_build.py`            → `rnd.pt` (trains the RND predictor MLP)
2. `build_rho_reference.py`  → `rho_reference.npz` (PCA reference bank, for the kNN baseline)
3. `precompute_dimer_table.py` → dimer residual tables

**Calibrate / tune the gate:**
- `calibrate_anchor.py`, `tune_gate.py`, `tune_lambda.py`, `tune_pairphys.py`,
  `tune_selective.py` (SelectiveNet-style risk–coverage threshold).

**Evaluate & compare:**
- `clean_split_eval.py`, `compare_all.py`, `compare_anchor_vs_ft.py`,
  `compare_multihead.py`, `keep_3way.py`, `keep_compare_final.py`, `error_map.py`,
  `extended_highP.py`, `mlip_arena_diatomics.py`, `mlip_arena_highP.py`.
- `bench_fix.py` / `bench_overhead.py` — wall-clock overhead of the anchor.
- `bench_mechanical_stability.py`, `pka_radiation.py`, `radiation_2mev.py`.

**Figures:** `make_*_figures.py`, `make_rnd_*.py` regenerate the plots.

## `md_stability/` — MD short-range stability
`scripts/anchor_calculator.py` is the reusable `AnchorCalculator` (an ASE
calculator wrapping MACE + the anchor; it is also imported by the LAMMPS driver
in `../lammps/`). `md_run.py`, `dimer_collision.py`, `dimer_scan.py`,
`select_systems.py`, `analyze.py`, `plot_scan.py` run and analyze the stability
suite.

## `raddmg/` — radiation-damage validation
Static defect energetics, threshold-displacement / recoil sweeps, SrTiO₃ Frenkel
pairs, Wigner–Seitz occupation analysis, and MLIP-vs-VASP force checks
(`vasp_forces/scripts/`). `scripts/common_calc.py` is the shared calculator
factory. Entry points: `run_defects_check.sh`, `run_srtio3_check.sh`.

## Required environment variables
`ZBL_MACE_MH0` (base model), `ZBL_ANCHOR_RESULTS` (artifact dir),
`ZBL_EVAL_DATA` (VASP splits), `ZBL_MPTRJ_XYZ` (in-distribution bank). See
`config.example.env`.
