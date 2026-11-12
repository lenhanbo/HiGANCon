#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=""
CHECKPOINT_LINK=""
CHECKPOINT_FILENAME="latest.pt"
HF_REVISION="main"
RUN_MODE="contrastive"
MANIFEST_ROOT="/kaggle/working/data"
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
  bash scripts/kaggle_eval_hf.sh --dataset-root PATH --checkpoint-link HF_SOURCE [options]

HF_SOURCE accepts:
  https://huggingface.co/owner/repo/resolve/main/latest.pt
  https://huggingface.co/owner/repo/blob/main/latest.pt
  hf://owner/repo/latest.pt@main
  owner/repo                       (uses --checkpoint-filename latest.pt)

Options:
  --checkpoint-filename FILE       Used with owner/repo (default: latest.pt)
  --hf-revision REVISION           Used with owner/repo or hf:// without @revision
  --run-mode baseline|contrastive
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
    --run-mode) RUN_MODE="${2-}"; shift 2 ;;
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

if [[ -z "$DATASET_ROOT" || -z "$CHECKPOINT_LINK" ]]; then
  echo "--dataset-root and --checkpoint-link are required" >&2
  usage >&2
  exit 2
fi
RUN_MODE="${RUN_MODE,,}"
FID_KID_PROTOCOL="${FID_KID_PROTOCOL,,}"
PREVIEW_QUERY_MODE="${PREVIEW_QUERY_MODE,,}"
if [[ "$RUN_MODE" != "baseline" && "$RUN_MODE" != "contrastive" ]]; then
  echo "--run-mode must be baseline or contrastive" >&2
  exit 2
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

if [[ ! -f "$MANIFEST_ROOT/test.jsonl" || ! -f "$MANIFEST_ROOT/split_metadata.json" ]]; then
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

tar -czf "$OUTPUT_ROOT/evaluation_bundle.tar.gz" -C "$OUTPUT_ROOT" \
  evaluation.json evaluation_previews
printf 'Metrics: %s\n' "$OUTPUT_ROOT/evaluation.json"
printf 'Preview gallery: %s\n' "$OUTPUT_ROOT/evaluation_previews"
printf 'Bundle: %s\n' "$OUTPUT_ROOT/evaluation_bundle.tar.gz"
