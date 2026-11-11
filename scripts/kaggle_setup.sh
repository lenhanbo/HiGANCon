#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:?Usage: bash scripts/kaggle_setup.sh DATASET_ROOT}"
test -f "$DATASET_ROOT/forms.txt"
test -f "$DATASET_ROOT/linux_gt.txt"
test -f "$DATASET_ROOT/train_gt.txt"
test -f "$DATASET_ROOT/val_gt.txt"
test -d "$DATASET_ROOT/words"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("A Kaggle GPU accelerator is required")
PY

python -m pip install -q -r requirements-kaggle.txt
python prepare_iam.py --dataset-root "$DATASET_ROOT" --output /kaggle/working/data --seed 42
pytest -q
