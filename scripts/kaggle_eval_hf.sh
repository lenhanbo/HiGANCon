#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=""
DATASET_FORMAT="higan_hdf5"
CHECKPOINT_LINK=""
CHECKPOINT_FILENAME="latest.pt"
HF_REVISION="main"
HDF5_PREFIX=""
RUN_MODE="contrastive"
MANIFEST_ROOT=""
OUTPUT_ROOT=""
EVAL_MAX_SAMPLES="0"
PREVIEW_SAMPLES="64"
PREVIEW_PAGE_SIZE="16"
PREVIEW_QUERY_MODE="random"
PREVIEW_QUERY_TEXT=""
FID_KID_PROTOCOL="higan"
EVAL_SEED="123456"
RECOGNIZER_CHECKPOINT=""
HWD_CHECKPOINT=""
SKIP_HWD="true"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/kaggle_eval_hf.sh --checkpoint-link HF_SOURCE [options]

HF_SOURCE accepts:
  https://huggingface.co/owner/repo/resolve/main/latest.pt
  https://huggingface.co/owner/repo/blob/main/latest.pt
  hf://owner/repo/latest.pt@main
  owner/repo                       (uses --checkpoint-filename latest.pt)

Options:
  --checkpoint-filename FILE       Used with owner/repo (default: latest.pt)
  --hf-revision REVISION           Used with owner/repo or hf:// without @revision
  --dataset-root PATH               Optional for HDF5; downloads from checkpoint repo if omitted
  --hdf5-prefix FOLDER              Hub folder containing the four HDF5 files
  --run-mode baseline|contrastive
  --dataset-format iam_raw|higan_hdf5
  --manifest-root PATH
  --output-root PATH
  --eval-max-samples COUNT         0 evaluates all test samples
  --preview-samples COUNT          64 default; -1 saves every evaluated sample; 0 disables
  --preview-page-size COUNT
  --preview-query-mode dataset|random
  --preview-query-text TEXT         Fixed query for every reference
  --fid-kid-protocol higan|torchvision
  --eval-seed SEED
  --recognizer-checkpoint PATH
  --with-hwd
  --hwd-checkpoint PATH_OR_HF_URI

For a private Hub repository, set HF_TOKEN in Kaggle Secrets before running.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="${2-}"; shift 2 ;;
    --checkpoint-link) CHECKPOINT_LINK="${2-}"; shift 2 ;;
    --checkpoint-filename) CHECKPOINT_FILENAME="${2-}"; shift 2 ;;
    --hf-revision) HF_REVISION="${2-}"; shift 2 ;;
    --hdf5-prefix) HDF5_PREFIX="${2-}"; shift 2 ;;
    --run-mode) RUN_MODE="${2-}"; shift 2 ;;
    --dataset-format) DATASET_FORMAT="${2-}"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="${2-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2-}"; shift 2 ;;
    --eval-max-samples) EVAL_MAX_SAMPLES="${2-}"; shift 2 ;;
    --preview-samples) PREVIEW_SAMPLES="${2-}"; shift 2 ;;
    --preview-page-size) PREVIEW_PAGE_SIZE="${2-}"; shift 2 ;;
    --preview-query-mode) PREVIEW_QUERY_MODE="${2-}"; shift 2 ;;
    --preview-query-text) PREVIEW_QUERY_TEXT="${2-}"; shift 2 ;;
    --fid-kid-protocol) FID_KID_PROTOCOL="${2-}"; shift 2 ;;
    --eval-seed) EVAL_SEED="${2-}"; shift 2 ;;
    --recognizer-checkpoint) RECOGNIZER_CHECKPOINT="${2-}"; shift 2 ;;
    --with-hwd) SKIP_HWD="false"; shift ;;
    --hwd-checkpoint) HWD_CHECKPOINT="${2-}"; SKIP_HWD="false"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$CHECKPOINT_LINK" ]]; then
  echo "--checkpoint-link is required" >&2
  usage >&2
  exit 2
fi
RUN_MODE="${RUN_MODE,,}"
DATASET_FORMAT="${DATASET_FORMAT,,}"
FID_KID_PROTOCOL="${FID_KID_PROTOCOL,,}"
PREVIEW_QUERY_MODE="${PREVIEW_QUERY_MODE,,}"
if [[ "$RUN_MODE" != "baseline" && "$RUN_MODE" != "contrastive" ]]; then
  echo "--run-mode must be baseline or contrastive" >&2
  exit 2
fi
if [[ "$DATASET_FORMAT" != "iam_raw" && "$DATASET_FORMAT" != "higan_hdf5" ]]; then
  echo "--dataset-format must be iam_raw or higan_hdf5" >&2
  exit 2
fi
if [[ "$DATASET_FORMAT" == "iam_raw" && -z "$DATASET_ROOT" ]]; then
  echo "--dataset-root is required for iam_raw" >&2
  exit 2
fi
if [[ -z "$MANIFEST_ROOT" ]]; then
  if [[ "$DATASET_FORMAT" == "higan_hdf5" ]]; then
    MANIFEST_ROOT="/kaggle/working/data_higan_hdf5"
  else
    MANIFEST_ROOT="/kaggle/working/data"
  fi
fi
if [[ "$FID_KID_PROTOCOL" != "higan" && "$FID_KID_PROTOCOL" != "torchvision" ]]; then
  echo "--fid-kid-protocol must be higan or torchvision" >&2
  exit 2
fi
if [[ "$PREVIEW_QUERY_MODE" != "dataset" && "$PREVIEW_QUERY_MODE" != "random" ]]; then
  echo "--preview-query-mode must be dataset or random" >&2
  exit 2
fi

CONFIG="configs/kaggle_${RUN_MODE}.yaml"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="/kaggle/working/hf_evaluations/${RUN_MODE}"
fi
mkdir -p "$OUTPUT_ROOT/evaluation_previews" /kaggle/working/hf_checkpoints

if [[ "$DATASET_FORMAT" == "higan_hdf5" && -z "$DATASET_ROOT" ]]; then
  DATASET_ROOT="$(python scripts/download_hf_hdf5.py "$CHECKPOINT_LINK" \
    --revision "$HF_REVISION" --prefix "$HDF5_PREFIX" \
    --output-dir /kaggle/working/higan_hdf5)"
  printf 'Downloaded HiGAN HDF5 root: %s\n' "$DATASET_ROOT"
fi

if [[ "$DATASET_FORMAT" == "higan_hdf5" ]]; then
  python prepare_higan_hdf5.py --hdf5-root "$DATASET_ROOT" --output "$MANIFEST_ROOT"
elif [[ ! -f "$MANIFEST_ROOT/split_metadata.json" ]]; then
    python prepare_iam.py --dataset-root "$DATASET_ROOT" --output "$MANIFEST_ROOT" --seed 42
fi

CHECKPOINT_PATH="$(python scripts/download_hf_checkpoint.py "$CHECKPOINT_LINK" \
  --filename "$CHECKPOINT_FILENAME" --revision "$HF_REVISION" \
  --output-dir /kaggle/working/hf_checkpoints)"
printf 'Downloaded checkpoint: %s\n' "$CHECKPOINT_PATH"

recognizer_args=()
if [[ -n "$RECOGNIZER_CHECKPOINT" ]]; then
  recognizer_args=(--recognizer-checkpoint "$RECOGNIZER_CHECKPOINT")
fi
hwd_args=(--skip-hwd)
if [[ "$SKIP_HWD" == "false" ]]; then
  hwd_args=()
  if [[ -n "$HWD_CHECKPOINT" ]]; then
    hwd_args=(--hwd-checkpoint "$HWD_CHECKPOINT")
  fi
fi
preview_query_args=(--preview-query-mode "$PREVIEW_QUERY_MODE")
if [[ -n "$PREVIEW_QUERY_TEXT" ]]; then
  preview_query_args+=(--preview-query-text "$PREVIEW_QUERY_TEXT")
fi

python evaluate.py \
  --config "$CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --dataset-format "$DATASET_FORMAT" \
  --manifest-root "$MANIFEST_ROOT" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output "$OUTPUT_ROOT/evaluation.json" \
  --max-samples "$EVAL_MAX_SAMPLES" \
  --fid-kid-protocol "$FID_KID_PROTOCOL" \
  --eval-seed "$EVAL_SEED" \
  --preview-samples "$PREVIEW_SAMPLES" \
  --preview-page-size "$PREVIEW_PAGE_SIZE" \
  --preview-dir "$OUTPUT_ROOT/evaluation_previews" \
  "${preview_query_args[@]}" "${recognizer_args[@]}" "${hwd_args[@]}"

printf 'Metrics: %s\n' "$OUTPUT_ROOT/evaluation.json"
printf 'Preview gallery: %s\n' "$OUTPUT_ROOT/evaluation_previews"
