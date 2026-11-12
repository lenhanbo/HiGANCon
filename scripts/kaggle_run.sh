#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=""
RUN_MODE="contrastive"
RESUME_PATH=""
RECOGNIZER_CHECKPOINT=""
MAX_HOURS="11.5"
HWD_CHECKPOINT=""
EVAL_MAX_SAMPLES="0"
FID_KID_PROTOCOL="higan"
EVAL_SEED="123456"
CONTRASTIVE_START_STEP="28841"
CONTRASTIVE_RAMP_STEPS="28840"
MAX_STEPS="0"
SKIP_HWD="false"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/kaggle_run.sh --dataset-root PATH [options]

Options:
  --run-mode baseline|contrastive
  --resume PATH
  --recognizer-checkpoint PATH
  --max-hours HOURS
  --max-steps STEPS                 0 means unlimited
  --contrastive-start-step STEP
  --contrastive-ramp-steps STEPS
  --eval-max-samples COUNT          0 means the full test manifest
  --fid-kid-protocol higan|torchvision
  --eval-seed SEED
  --hwd-checkpoint PATH_OR_HF_URI
  --skip-hwd

Legacy positional arguments remain supported for existing notebooks.
EOF
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  # Backward-compatible positional interface.
  DATASET_ROOT="$1"
  RUN_MODE="${2:-contrastive}"
  RESUME_PATH="${3:-}"
  RECOGNIZER_CHECKPOINT="${4:-}"
  MAX_HOURS="${5:-11.5}"
  HWD_CHECKPOINT="${6:-}"
  EVAL_MAX_SAMPLES="${7:-0}"
  CONTRASTIVE_START_STEP="${8:-28841}"
  CONTRASTIVE_RAMP_STEPS="${9:-28840}"
  MAX_STEPS="${10:-0}"
  SKIP_HWD="${11:-false}"
  FID_KID_PROTOCOL="${12:-higan}"
  EVAL_SEED="${13:-123456}"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-root) DATASET_ROOT="${2-}"; shift 2 ;;
      --run-mode) RUN_MODE="${2-}"; shift 2 ;;
      --resume) RESUME_PATH="${2-}"; shift 2 ;;
      --recognizer-checkpoint) RECOGNIZER_CHECKPOINT="${2-}"; shift 2 ;;
      --max-hours) MAX_HOURS="${2-}"; shift 2 ;;
      --max-steps) MAX_STEPS="${2-}"; shift 2 ;;
      --contrastive-start-step) CONTRASTIVE_START_STEP="${2-}"; shift 2 ;;
      --contrastive-ramp-steps) CONTRASTIVE_RAMP_STEPS="${2-}"; shift 2 ;;
      --eval-max-samples) EVAL_MAX_SAMPLES="${2-}"; shift 2 ;;
      --fid-kid-protocol) FID_KID_PROTOCOL="${2-}"; shift 2 ;;
      --eval-seed) EVAL_SEED="${2-}"; shift 2 ;;
      --hwd-checkpoint) HWD_CHECKPOINT="${2-}"; shift 2 ;;
      --skip-hwd) SKIP_HWD="true"; shift ;;
      --help|-h) usage; exit 0 ;;
      *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
  done
fi

if [[ -z "$DATASET_ROOT" ]]; then
  echo "--dataset-root is required" >&2
  usage >&2
  exit 2
fi

RUN_MODE="${RUN_MODE,,}"

if [[ "$RUN_MODE" != "baseline" && "$RUN_MODE" != "contrastive" ]]; then
  printf 'RUN_MODE must be baseline or contrastive; received <%q>\n' "$RUN_MODE" >&2
  exit 2
fi
FID_KID_PROTOCOL="${FID_KID_PROTOCOL,,}"
if [[ "$FID_KID_PROTOCOL" != "higan" && "$FID_KID_PROTOCOL" != "torchvision" ]]; then
  printf 'FID_KID_PROTOCOL must be higan or torchvision; received <%q>\n' "$FID_KID_PROTOCOL" >&2
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
  --fid-kid-protocol "$FID_KID_PROTOCOL" --eval-seed "$EVAL_SEED" \
  "${recognizer_args[@]}" "${hwd_args[@]}"

tar -czf /kaggle/working/run_bundle.tar.gz -C /kaggle/working data outputs
