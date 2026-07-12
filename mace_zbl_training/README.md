# mace_zbl_training/ — ZBL-MACE fine-tuning & evaluation pipeline

A numbered pipeline for the MACE side: download the foundation models, run
zero-shot predictions, evaluate, optionally fine-tune, and produce comparison
figures. This is where the base/fine-tuned MACE checkpoints used by `../anchor/`
come from.

## `scripts/` (run in order)
1. `01_download_models.py`   — fetch MACE-MH-0 / MH-1 foundation weights into your
   local model directory (reuses an existing copy via symlink if present).
2. `02_predict.py`           — zero-shot energy/force predictions on the eval sets.
3. `03_evaluate.py`          — metrics (MAE / RMSE / R², per-element shifts).
4. `04_finetune.py`          — optional readout/scale-shift fine-tuning.
5. `05_distribution_compare.py` — compare error distributions across models.
6. `06_summary_figure.py`    — summary deck figure.
- `plot_training_curves.py`, `plot_deck_extras.py` — auxiliary plots.

`env.yml` is a reference for the conda environment used (MACE-torch + ASE +
PyTorch). Prefer the repo-root `environment.yml`.

## Required environment variables
`ZBL_MACE_ZBL_MODEL` / `ZBL_MACE_MH0` / `ZBL_MACE_MH1` (model paths),
`ZBL_EVAL_DATA`, `ZBL_MPTRJ_XYZ`, `ZBL_MIXED_DATA`. See `config.example.env`.
Model weights and datasets are **not** shipped in this repo.
