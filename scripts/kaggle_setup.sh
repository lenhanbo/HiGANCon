#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:?Usage: bash scripts/kaggle_setup.sh DATASET_ROOT [iam_raw|higan_hdf5] [MANIFEST_ROOT]}"
DATASET_FORMAT="${2:-iam_raw}"
if [[ "$DATASET_FORMAT" == "higan_hdf5" ]]; then
  MANIFEST_ROOT="${3:-/kaggle/working/data_higan_hdf5}"
else
  MANIFEST_ROOT="${3:-/kaggle/working/data}"
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("A Kaggle GPU accelerator is required")
PY

python -m pip install -q -r requirements-kaggle.txt
if [[ "$DATASET_FORMAT" == "iam_raw" ]]; then
  python prepare_iam.py --dataset-root "$DATASET_ROOT" --output "$MANIFEST_ROOT" --seed 42
elif [[ "$DATASET_FORMAT" == "higan_hdf5" ]]; then
  python prepare_higan_hdf5.py --hdf5-root "$DATASET_ROOT" --output "$MANIFEST_ROOT"
else
  echo "DATASET_FORMAT must be iam_raw or higan_hdf5" >&2
  exit 2
fi
pytest -q
