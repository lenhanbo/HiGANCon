#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=""
DATASET_FORMAT="iam_raw"
MANIFEST_ROOT=""
RUN_MODE="contrastive"
SUPCON_TEXT_MODE="per_item"
RESUME_PATH=""
RECOGNIZER_CHECKPOINT=""
MAX_HOURS="11.5"
HWD_CHECKPOINT=""
EVAL_MAX_SAMPLES="0"
FID_KID_PROTOCOL="higan"
EVAL_SEED="123456"
MAX_STEPS="0"
SKIP_HWD="false"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/kaggle_run.sh --dataset-root PATH [options]

Options:
  --run-mode baseline|supcon|contrastive|ctc|full
  --supcon-shared-text             Use one shared SupCon query text
  --supcon-per-item-text           Use a separate query text per SupCon item (default)
  --dataset-format iam_raw|higan_hdf5
  --manifest-root PATH
  --resume PATH
  --recognizer-checkpoint PATH
  --max-hours HOURS
  --max-steps STEPS                 0 means unlimited
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
  MAX_STEPS="${10:-0}"
  SKIP_HWD="${11:-false}"
  FID_KID_PROTOCOL="${12:-higan}"
  EVAL_SEED="${13:-123456}"
  DATASET_FORMAT="${14:-iam_raw}"
  MANIFEST_ROOT="${15:-}"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-root) DATASET_ROOT="${2-}"; shift 2 ;;
      --run-mode) RUN_MODE="${2-}"; shift 2 ;;
      --supcon-shared-text) SUPCON_TEXT_MODE="shared"; shift ;;
      --supcon-per-item-text) SUPCON_TEXT_MODE="per_item"; shift ;;
      --dataset-format) DATASET_FORMAT="${2-}"; shift 2 ;;
      --manifest-root) MANIFEST_ROOT="${2-}"; shift 2 ;;
      --resume) RESUME_PATH="${2-}"; shift 2 ;;
      --recognizer-checkpoint) RECOGNIZER_CHECKPOINT="${2-}"; shift 2 ;;
      --max-hours) MAX_HOURS="${2-}"; shift 2 ;;
      --max-steps) MAX_STEPS="${2-}"; shift 2 ;;
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
DATASET_FORMAT="${DATASET_FORMAT,,}"
SUPCON_TEXT_MODE="${SUPCON_TEXT_MODE,,}"

if [[ "$RUN_MODE" != "baseline" && "$RUN_MODE" != "supcon" && \
      "$RUN_MODE" != "contrastive" && "$RUN_MODE" != "ctc" && "$RUN_MODE" != "full" ]]; then
  printf 'RUN_MODE must be baseline, supcon, contrastive, ctc, or full; received <%q>\n' "$RUN_MODE" >&2
  exit 2
fi
if [[ "$DATASET_FORMAT" != "iam_raw" && "$DATASET_FORMAT" != "higan_hdf5" ]]; then
  echo "DATASET_FORMAT must be iam_raw or higan_hdf5" >&2
  exit 2
fi
if [[ "$SUPCON_TEXT_MODE" != "shared" && "$SUPCON_TEXT_MODE" != "per_item" ]]; then
  printf 'SUPCON_TEXT_MODE must be shared or per_item; received <%q>\n' "$SUPCON_TEXT_MODE" >&2
  exit 2
fi
if [[ -z "$MANIFEST_ROOT" ]]; then
  if [[ "$DATASET_FORMAT" == "higan_hdf5" ]]; then
    MANIFEST_ROOT="/kaggle/working/data_higan_hdf5"
  else
    MANIFEST_ROOT="/kaggle/working/data"
  fi
fi
FID_KID_PROTOCOL="${FID_KID_PROTOCOL,,}"
if [[ "$FID_KID_PROTOCOL" != "higan" && "$FID_KID_PROTOCOL" != "torchvision" ]]; then
  printf 'FID_KID_PROTOCOL must be higan or torchvision; received <%q>\n' "$FID_KID_PROTOCOL" >&2
  exit 2
fi
CONFIG="configs/config.yaml"
GENERATOR_VARIANT="$(python -c 'import yaml; c=yaml.safe_load(open("configs/config.yaml")); print(c.get("architecture", {}).get("generator_variant", "baseline"))')"
if [[ "$GENERATOR_VARIANT" != "baseline" && "$GENERATOR_VARIANT" != "multiscale_condition" ]]; then
  printf 'architecture.generator_variant must be baseline or multiscale_condition; received <%q>\n' "$GENERATOR_VARIANT" >&2
  exit 2
fi
OUTPUT_NAME="$RUN_MODE"
if [[ "$RUN_MODE" == "supcon" || "$RUN_MODE" == "contrastive" || "$RUN_MODE" == "full" ]]; then
  OUTPUT_NAME="${RUN_MODE}-${SUPCON_TEXT_MODE}"
fi
if [[ "$GENERATOR_VARIANT" != "baseline" ]]; then
  OUTPUT_NAME="${OUTPUT_NAME}-${GENERATOR_VARIANT}"
fi
OUTPUT="/kaggle/working/outputs/${OUTPUT_NAME}"

loss_args=()
case "$RUN_MODE" in
  baseline) loss_args=(--no supcon --no ctc-guided) ;;
  supcon|contrastive) loss_args=(--no ctc-guided) ;;
  ctc) loss_args=(--no supcon) ;;
  full) loss_args=() ;;
esac

if [[ "$SUPCON_TEXT_MODE" == "shared" ]]; then
  supcon_text_args=(--supcon-shared-text)
else
  supcon_text_args=(--supcon-per-item-text)
fi

resume_args=()
if [[ -n "$RESUME_PATH" ]]; then
  resume_args=(--resume "$RESUME_PATH")
fi

python train.py --config "$CONFIG" --dataset-root "$DATASET_ROOT" \
  --dataset-format "$DATASET_FORMAT" --manifest-root "$MANIFEST_ROOT" \
  --output "$OUTPUT" --max-hours "$MAX_HOURS" \
  --max-steps "$MAX_STEPS" \
  --no-eval \
  "${resume_args[@]}" \
  "${loss_args[@]}" \
  "${supcon_text_args[@]}"

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
  --dataset-format "$DATASET_FORMAT" --manifest-root "$MANIFEST_ROOT" \
  --checkpoint "$OUTPUT/checkpoints/last.pt" \
  --output "$OUTPUT/evaluation.json" --max-samples "$EVAL_MAX_SAMPLES" \
  --fid-kid-protocol "$FID_KID_PROTOCOL" --eval-seed "$EVAL_SEED" \
  "${recognizer_args[@]}" "${hwd_args[@]}"
