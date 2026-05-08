# Repository Guidelines

## Project Structure & Module Organization
Core forecasting architectures live in `model/` (Lite, iTransformer, FourierGNN, TimeMixer, etc.), each exposing a `Model` class consumed by `experiments/exp_long_term_forecasting.py`. Shared attention, gating, and low-rank blocks sit in `layers/`, while dataset readers plus preprocessing steps are under `data_provider/` and `data_preprocess/`. Raw CSVs belong in `dataset/`, derived tensors or checkpoints in `checkpoints/`, and published metrics or plots in `results/`, `frequency_outputs/`, or `visual/`. Curated launchers under `scripts/` capture tuned hyperparameters; keep ad-hoc notebooks out of git.

## Build, Test, and Development Commands
Create the recommended environment with `conda env create -f environment.yml && conda activate fcgformer`, or reuse an existing virtualenv via `python -m pip install -r requirements.txt`. Typical training run: `python run.py --is_training 1 --model Lite --model_id etth1_lite --data ETTh1 --root_path ./dataset/ETT-small/ --seq_len 96 --pred_len 24`. Evaluate a saved checkpoint by toggling `--is_training 0` and pointing `--checkpoints` at the artifact folder. Lite regression changes should pass `python test_lite_compatibility.py` and `python test_optimizations.py`; archive console output under `test_results/`.

## Coding Style & Naming Conventions
Target Python 3.9+, use 4‑space indentation, and follow PEP 8 with snake_case identifiers plus CamelCase filenames for concrete model variants (`Lite_v5_mobile.py`). Include type hints on new public utilities so they interoperate with the experiment runner. Reuse existing logging (plain `print` for experiments, `logging` inside libraries) and route configuration through the shared ArgParser rather than module-level globals.

## Testing Guidelines
Regression coverage relies on executable scripts; add new probes as `test_<capability>.py` in the repo root so they run via `python test_new_feature.py`. Guard GPU-only paths with `torch.cuda.is_available()` and fall back to CPU to keep CI lightweight. When touching data loaders, run a short experiment (`--itr 1 --epochs 1`) to ensure `Exp_Long_Term_Forecast.train` still completes, and share the resulting metrics table in your PR. Store heavy artifacts or FLOP dumps in `test_results/` or `light_result/` instead of committing binaries.

## Commit & Pull Request Guidelines
History favors concise, imperative subjects (often in Chinese) such as `fgnn模型修改` or `结果排版优化`; mirror that style and keep each commit focused on one change. Mention the affected model or dataset in the subject and elaborate only when the rationale is non-obvious. Pull requests should summarize the scenario, list reproducible commands, link the relevant log from `results/` or `frequency_outputs/`, and note any dependency or hardware implications (update `requirements.txt`/`environment.yml` when needed). Provide screenshots for visualization tweaks and flag security-sensitive configs so reviewers can scrub them before merge.

## 用中文回答