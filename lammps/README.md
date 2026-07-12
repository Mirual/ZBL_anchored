# lammps/ — deploying MACE / DPA-3.1 / anchor potentials in LAMMPS

Runs all four potentials (MACE, MACE-anchor, DPA-3.1, DPA-anchor) inside LAMMPS
on GPU, numerically matched to the ASE reference path. The core mechanism is a
Python-driven LAMMPS run using `fix external pf/callback`: at every step LAMMPS
hands out coordinates and the GPU calculator returns energy / forces / virial.

## `scripts/`
- `mlip_fixext.py` — **the driver**. One MPI rank / one GPU, whole cell per step,
  bit-identical to the ASE evaluation. Selects the calculator via `--calc`
  (`mace`, `mace-anchor`, `dpa`, `dpa-anchor`). Imports the `AnchorCalculator`
  from `../anchor/md_stability/scripts/`.
- `ase_md_baseline.py`, `gpu_mem_profile.py`, `max_cell_probe.py` — profilers that
  `import mlip_fixext` (kept in the same directory so the import resolves).
- `make_anchor_lite.py` — packs a lightweight anchor cache for deployment.
- `convert_dpa_borderop.py` — re-freezes a DPA-3.1 model with the customised
  border op so LAMMPS' native `pair_style deepmd` can load it.
- `build_dp_devel.sh`, `fix_cuda_targets.cmake` — build helpers for a DeePMD-kit
  LAMMPS with matching torch ABI.
- `build_bench_table.py` — aggregate the LAMMPS benchmark JSON into a table.
- `xyz2data.py` — convert an `.xyz` structure to a LAMMPS data file.
- `bench_lammps.sbatch` — SLURM benchmark driver.

## `mace_smoke/`, `dpa_smoke/` — numerical reference smoke tests
`ref_*.py` produce the ASE reference JSON; the `in.*` decks run the same system
through LAMMPS; `compare_dpa.py` checks agreement within tolerance. The input
data files (`data_*.lammps`, `*.npz`) are **not** committed — regenerate them
with `xyz2data.py` / `dpa_smoke/gen_system.py`.

## `examples/`
`cmds_nvt_dump.in` and `run_md.sbatch` are ready-to-edit templates for an NVT run
with trajectory dump.

## Required environment variables
`ZBL_MACE_MH1`, `ZBL_DPA_MODEL`, `ZBL_DPA_MODEL_BORDEROP`, `ZBL_ANCHOR_RESULTS`,
`ZBL_DEEPMD_PYTHON` (a deepmd-enabled interpreter for the native DPA path). See
`config.example.env`. Requires a LAMMPS build with the MLIAP-unified + Python
packages (and DeePMD-kit for the DPA path).
