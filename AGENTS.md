# HiGAN-Contrastive project conventions

This file is the source of truth for future coding sessions. Keep the training surface
simple and do not reintroduce parallel entry points or mode-specific YAML files.

## Canonical entry points and configuration

- Training always starts from `train.py`. Do not create `train_ctc_soft.py` or another
  training entry point.
- The only configuration file is `configs/config.yaml`. It contains model, optimizer,
  SupCon, CTC-guided matching, schedule, preview, and automatic-evaluation settings.
- Loss weights, start steps, ramp steps, and preview cadence are YAML-owned for the
  unified config. Do not add CLI schedule overrides for this config.
- `train.py` automatically evaluates `checkpoints/last.pt` after a completed training
  run. `--smoke-test` skips evaluation and `--no-eval` is the explicit opt-out.

## Loss modes

SupCon and CTC-guided matching are enabled by default. Keep both independent opt-outs:

- `--no supcon` or `--no-supcon`
- `--no ctc-guided` or `--no-ctc-guided`

The supported combinations are:

- `full`: no loss opt-outs (HiGAN + SupCon + CTC-guided)
- `supcon` / legacy alias `contrastive`: `--no ctc-guided`
- `ctc`: `--no supcon`
- `baseline`: `--no supcon --no ctc-guided`

## SupCon text modes

SupCon always uses one shared auxiliary noise vector across the batch. Its query text is
configurable without code changes:

- `--supcon-shared-text`: all SupCon views use the first query text.
- `--supcon-per-item-text`: every SupCon view uses its own query text.

The flags are mutually exclusive. The default is
`contrastive.text_mode: per_item` in `configs/config.yaml`. Store the selected mode in
run metadata/checkpoints and reject resume when it does not match.

## Kaggle and GitHub workflow

- Kaggle clones/pulls the GitHub repository, then runs `scripts/kaggle_run.sh`.
- `scripts/kaggle_run.sh` must use `configs/config.yaml` and pass the chosen loss mode
  and SupCon text flag through to `train.py`.
- The script passes `--no-eval` to `train.py` only because it invokes `evaluate.py`
  itself with Kaggle-specific metric options; it must still evaluate exactly once.
- The Kaggle notebook exposes `RUN_MODE` (`baseline`, `supcon`, `contrastive`, `ctc`, or
  `full`) and `SUPCON_TEXT_MODE` (`shared` or `per_item`). Keep notebook, shell script,
  README, and CLI names synchronized.
- After behavior changes, run compile checks and the complete pytest suite.
