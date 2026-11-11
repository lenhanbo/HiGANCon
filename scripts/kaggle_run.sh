#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:?Usage: bash scripts/kaggle_run.sh DATASET_ROOT [baseline|contrastive] [RESUME_PATH] [RECOGNIZER_CHECKPOINT] [MAX_HOURS] [HWD_CHECKPOINT] [EVAL_MAX_SAMPLES]}"
RUN_MODE="${2:-contrastive}"
RESUME_PATH="${3:-}"
RECOGNIZER_CHECKPOINT="${4:-}"
MAX_HOURS="${5:-11.5}"
HWD_CHECKPOINT="${6:-}"
EVAL_MAX_SAMPLES="${7:-512}"

if [[ "$RUN_MODE" != "baseline" && "$RUN_MODE" != "contrastive" ]]; then
  echo "RUN_MODE must be baseline or contrastive" >&2
  exit 2
fi
CONFIG="configs/kaggle_${RUN_MODE}.yaml"
OUTPUT="/kaggle/working/outputs/${RUN_MODE}"

resume_args=()
if [[ -n "$RESUME_PATH" ]]; then
  resume_args=(--resume "$RESUME_PATH")
fi

python train.py --config "$CONFIG" --dataset-root "$DATASET_ROOT" \
  --manifest-root /kaggle/working/data --output "$OUTPUT" --max-hours "$MAX_HOURS" "${resume_args[@]}"

recognizer_args=()
if [[ -n "$RECOGNIZER_CHECKPOINT" ]]; then
  recognizer_args=(--recognizer-checkpoint "$RECOGNIZER_CHECKPOINT")
fi
hwd_args=()
if [[ -n "$HWD_CHECKPOINT" ]]; then
  hwd_args=(--hwd-checkpoint "$HWD_CHECKPOINT")
fi
python evaluate.py --config "$CONFIG" --dataset-root "$DATASET_ROOT" \
  --manifest-root /kaggle/working/data --checkpoint "$OUTPUT/checkpoints/last.pt" \
  --output "$OUTPUT/evaluation.json" --max-samples "$EVAL_MAX_SAMPLES" \
  "${recognizer_args[@]}" "${hwd_args[@]}"

tar -czf /kaggle/working/run_bundle.tar.gz -C /kaggle/working data outputs
