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

## Generator architecture variants

- `architecture.generator_variant: baseline` must instantiate the exact original
  `networks.BigGAN_networks.Generator`; do not add parameters or alter its forward path.
- `architecture.generator_variant: multiscale_condition` enables the opt-in hierarchical
  reference encoder. It conditions only reference-guided generation; random `G(z, text)`
  generation must continue to bypass the reference encoder.
- Keep architecture selection YAML-owned. Do not add an architecture CLI flag or a second
  training entry point/config file.
- Reference-conditioned calls must pass `reference_lengths`, use padding-aware descriptors,
  and cache one encoded reference context when generating several query texts in evaluation.
- Preserve legacy baseline checkpoint loading. Reject resume/evaluation when a checkpoint's
  recorded generator variant differs from the configured variant.

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
- The canonical Kaggle notebook is `kaggle/higancon.ipynb`. Do not recreate or use the
  deleted `kaggle/HiGAN_Contrastive.ipynb` unless the user explicitly asks.
- The Kaggle notebook exposes `RUN_MODE` (`baseline`, `supcon`, `contrastive`, `ctc`, or
  `full`), `SUPCON_TEXT_MODE` (`shared` or `per_item`), and `GENERATOR_VARIANT`
  (`baseline` or `multiscale_condition`). Keep notebook, shell script, README, config,
  and CLI names synchronized.
- Preserve the current notebook style in later edits: one user-editable configuration
  cell, GitHub clone/pull, Hugging Face HDF5 download, dataset setup, mode-aware smoke
  test, long train plus exactly one eval, and a final results/preview cell. Modify this
  workflow in place instead of replacing it with a generic template.
- After behavior changes, run compile checks and the complete pytest suite.

## Text-recognition evaluation

- CTC greedy decoding must always be truncated to each sample's valid recognizer length;
  never decode padded timesteps.
- Report recognizer calibration on real test images plus generated-image CER/WER for
  same-text, different in-vocabulary text, and OOV text separately.
- Different-text queries come from train transcripts and must differ from the reference.
  OOV queries come from the open lexicon after excluding train transcripts.
- CER/WER are reportable only with one frozen recognizer trained independently on real
  data and reused across every compared generator. Without it, clearly label metrics as
  diagnostic rather than silently treating the jointly trained `R` as independent.
