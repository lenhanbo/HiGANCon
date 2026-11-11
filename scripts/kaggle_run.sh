#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:?Usage: bash scripts/kaggle_run.sh DATASET_ROOT [baseline|contrastive] [RESUME_PATH] [RECOGNIZER_CHECKPOINT] [MAX_HOURS] [HWD_CHECKPOINT] [EVAL_MAX_SAMPLES] [CONTRASTIVE_START_STEP] [CONTRASTIVE_RAMP_STEPS] [MAX_STEPS] [SKIP_HWD]}"
RUN_MODE="${2:-contrastive}"
RESUME_PATH="${3:-}"
RECOGNIZER_CHECKPOINT="${4:-}"
MAX_HOURS="${5:-11.5}"
HWD_CHECKPOINT="${6:-}"
EVAL_MAX_SAMPLES="${7:-512}"
CONTRASTIVE_START_STEP="${8:-28841}"
CONTRASTIVE_RAMP_STEPS="${9:-28840}"
MAX_STEPS="${10:-0}"
SKIP_HWD="${11:-false}"

if [[ "$RUN_MODE" != "baseline" && "$RUN_MODE" != "contrastive" ]]; then
  printf 'RUN_MODE must be baseline or contrastive; received argument 2 as <%q>\n' "$RUN_MODE" >&2
  exit 2
fi
CONFIG="configs/kaggle_${RUN_MODE}.yaml"
OUTPUT="/kaggle/working/outputs/${RUN_MODE}"

resume_args=()
if [[ -n "$RESUME_PATH" ]]; then
  resume_args=(--resume "$RESUME_PATH")
fi

python train.py --config "$CONFIG" --dataset-root "$DATASET_ROOT" \
  --manifest-root /kaggle/working/data --output "$OUTPUT" --max-hours "$MAX_HOURS" \
  --max-steps "$MAX_STEPS" \
  "${resume_args[@]}" \
  --contrastive-start-step "$CONTRASTIVE_START_STEP" \
  --contrastive-ramp-steps "$CONTRASTIVE_RAMP_STEPS"

recognizer_args=()
if [[ -n "$RECOGNIZER_CHECKPOINT" ]]; then
  recognizer_args=(--recognizer-checkpoint "$RECOGNIZER_CHECKPOINT")
fi
hwd_args=()
if [[ "$SKIP_HWD" == "true" || "$SKIP_HWD" == "1" ]]; then
  hwd_args=(--skip-hwd)
elif [[ -n "$HWD_CHECKPOINT" ]]; then
  hwd_args=(--hwd-checkpoint "$HWD_CHECKPOINT")
fi
python evaluate.py --config "$CONFIG" --dataset-root "$DATASET_ROOT" \
  --manifest-root /kaggle/working/data --checkpoint "$OUTPUT/checkpoints/last.pt" \
  --output "$OUTPUT/evaluation.json" --max-samples "$EVAL_MAX_SAMPLES" \
  "${recognizer_args[@]}" "${hwd_args[@]}"

tar -czf /kaggle/working/run_bundle.tar.gz -C /kaggle/working data outputs
