# dpa_variant/ — the physics anchor on top of DPA-3.1

The same uncertainty-gated physical anchor as `../anchor/`, but with the
**DPA-3.1** foundation model (via DeePMD-kit) as the base potential instead of
MACE. This demonstrates the method is architecture-agnostic.

## `scripts/`
- `dpa_common.py` — shared DPA loader (`load_dp`, `compute_descriptors`) and the
  dataset/split constants; imported by the other scripts.
- `gate.py` — the RND novelty gate (`RNDGate`, `mlp`) over DPA descriptors.
- `pair_physics.py` — per-pair ZBL / dimer-residual physics (`DimerCache`, `zbl_V`).
- `rnd_build.py` — trains the RND predictor for DPA descriptors.
- `predict.py` / `predict_zbl.py` — vanilla vs anchored DPA prediction.
- `tune_pairphys.py` — calibrate the per-pair residual strength.
- `compare.py`, `head_to_head_aligned.py` — DPA-vs-MACE head-to-head evaluation.
- `run_all.sh`, `run_rest.sh` — batch drivers.

## Required environment variables
`ZBL_DPA_MODEL` (DPA-3.1 checkpoint), plus `ZBL_EVAL_DATA` / `ZBL_MPTRJ_XYZ` and
`ZBL_ANCHOR_RESULTS` as in `../anchor/`. Requires a working **DeePMD-kit** install.
See `config.example.env`.
