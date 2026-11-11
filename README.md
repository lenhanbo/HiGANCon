# HiGAN-Contrastive

Standalone implementation of the original [HiGAN](https://github.com/ganji15/HiGAN)
training objective with one optional addition: generated-style supervised contrastive loss.
It is designed for reproducible one-shot writer-style experiments on Kaggle.

## What is preserved

The discriminator-side update trains `D`, `R`, and `W` with hinge adversarial loss,
real-image CTC, and real-image writer classification while `G/E` are frozen. The
generator-side update freezes `D/R/W` but propagates through them to generated images,
and retains both generated modes, fake CTC, latent style reconstruction, fake writer
classification, KL regularization, and HiGAN's adaptive gradient balancing.

The contrastive run adds only:

```text
lambda_con * SupCon(L2Norm(E_mu(G([shared_noise, E_mu(reference)], shared_text))))
```

Each batch has 8 unique writers and 2 distinct reference words per writer. Positive
pairs share writer identity; all other generated samples are negatives. The first five
steps before `start_step` use `lambda_con=0`; it then ramps linearly for `ramp_steps`
updates and holds 0.1. Both values are exposed in the Kaggle parameter cell.
The baseline configuration uses `lambda_con=0` and skips the entire extra forward graph.

## IAM layout and manifests

Only one dataset path is needed. It must point directly to:

```text
iam_dataset/
├── forms.txt
├── linux_gt.txt
├── train_gt.txt
├── val_gt.txt
└── words/a01/a01-000u/a01-000u-00-00.png
```

`train_gt.txt` and `val_gt.txt` are compatibility files, not experimental splits.
The preparation command reads all valid entries from `linux_gt.txt`, maps form to writer
through `forms.txt`, removes writers with fewer than two valid word images, shuffles
writers with seed 42, and writes deterministic 80/10/10 writer-disjoint manifests.

```bash
python prepare_iam.py --dataset-root "$DATASET_ROOT" --output /kaggle/working/data --seed 42
```

On Kaggle, the helper scripts take ordinary positional arguments; they do not require
environment variables:

```bash
bash scripts/kaggle_setup.sh "$DATASET_ROOT"
bash scripts/kaggle_run.sh "$DATASET_ROOT" "$RUN_MODE" "$RESUME_PATH" \
  "$RECOGNIZER_CHECKPOINT" "$MAX_HOURS" "$HWD_CHECKPOINT" "$EVAL_MAX_SAMPLES" \
  "$CONTRASTIVE_START_STEP" "$CONTRASTIVE_RAMP_STEPS" "$MAX_STEPS" \
  "$SKIP_HWD"
```

Images are grayscale, resized to height 32 with aspect ratio preserved, inverted and
normalized to `[-1,1]`, then padded with `-1` to a width divisible by 16.

## Training and resume

```bash
python train.py \
  --config configs/kaggle_contrastive.yaml \
  --dataset-root "$DATASET_ROOT" \
  --manifest-root /kaggle/working/data \
  --output /kaggle/working/outputs/contrastive \
  --max-hours 11.5
```

Use `configs/kaggle_baseline.yaml` for the paired baseline. Both use seed 123456,
AMP, Adam 2e-4 with betas (0.5, 0.999), 70 epochs, and HiGAN's linear decay. Their
`run_metadata.json` files must show the same `initial_state_hash` and `split_hash`.

Resume with the exact same config and manifests:

```bash
python train.py ... --resume /kaggle/input/previous-output/outputs/contrastive/checkpoints/last.pt
```

Checkpoints include `G/D/E/R/W`, optimizers, schedulers, AMP scaler, epoch/global step,
Python/NumPy/PyTorch/CUDA RNG, sampler position, Git commit, config/split hashes, and
initial-state hash. `--max-hours` saves `last.pt` before stopping.

`--max-steps N` is an exact global-step ceiling. It saves `last.pt` immediately after
step N; `0` disables this ceiling. On resume the restored global step is used, so raising
the ceiling continues training while lowering it below the saved step exits immediately.

Training saves a deterministic visual panel every 200 steps under `previews/`. It shows
`REAL REFERENCE | RANDOM SYNTHETIC STYLE | REFERENCE-GUIDED | SUPCON SHARED TEXT/NOISE`.
The random column uses one `z ~ N(0,I)` shared by every row, so multiple texts reveal one
synthetic style; it has no real writer identity. The adjacent JSON records its seed and
all 128 latent values. SupCon rows preserve sampler order, so each adjacent pair belongs
to one writer and uses the same text/noise. Preview generation uses a local RNG and eval
mode, so it does not advance training RNG or BatchNorm statistics. Change
`preview_every_steps` or `preview_items` in both experiment configs if needed.

Console loss records are averages over `print_every_steps` training batches. This is
important because D-side runs every batch while G/E-side runs every four batches; the
windowed record still includes `g_total`, `supcon`, and `lambda_con`. Per-batch values
remain in `metrics.jsonl`. For the contrastive config, `lambda_con` and `supcon` are zero
before `CONTRASTIVE_START_STEP`; the weight then rises per training step over
`CONTRASTIVE_RAMP_STEPS`. Set ramp to 0 or 1 to switch immediately to the target weight.

If batch 16 is too large, change `writers_per_batch` to 4 and `batch_size` to 8 in
**both** configs before starting either experiment. This gives 4 writers × 2 references.

## Evaluation

```bash
python evaluate.py \
  --config configs/kaggle_contrastive.yaml \
  --dataset-root "$DATASET_ROOT" \
  --manifest-root /kaggle/working/data \
  --checkpoint /kaggle/working/outputs/contrastive/checkpoints/last.pt \
  --recognizer-checkpoint /kaggle/input/independent-iam-recognizer/model.pt \
  --output /kaggle/working/outputs/contrastive/evaluation.json
```

The report contains FID, KID, HWD, CER, WER, writer retrieval Top-1/Top-5,
positive/negative cosine, separation gap, and noise diversity. HWD follows the official
BMVC 2023 protocol: Font²-pretrained VGG-16 features at height 32, writer-wise feature
means, Euclidean distance, then an average over writers. Lower FID, KID, and HWD are
better. The official HWD backbone downloads once and is cached by PyTorch; pass
`--hwd-checkpoint` to use an attached local checkpoint without Internet. It also accepts
`hf://owner/repo/file.pth@revision` and downloads through `huggingface_hub`. No official
HWD/Font² checkpoint was found on the Hub, so the empty default still uses the authors'
official GitHub release. Set `SKIP_HWD=true` (or pass `--skip-hwd` directly to
`evaluate.py`) to avoid downloading/loading the HWD backbone; the report records
`hwd: null` and `hwd_skipped: true`. For valid
independent CER/WER reporting,
supply an IAM recognizer trained outside both experiments; otherwise the report marks
`independent_recognizer: false`.

Success means improved writer retrieval and separation gap, CER degradation at most
1 percentage point, and KID degradation at most 5% versus baseline.

## Kaggle and tests

Open [`kaggle/HiGAN_Contrastive.ipynb`](kaggle/HiGAN_Contrastive.ipynb), set its variables,
and run all cells. It clones a pinned ref, validates CUDA and the explicit dataset root,
installs packages without replacing Kaggle's PyTorch, creates manifests, runs tests and a
one-batch smoke update, trains/resumes, evaluates, and creates `run_bundle.tar.gz`.

```bash
pytest -q
python -m compileall -q higan_contrastive prepare_iam.py train.py evaluate.py
```

IAM, large generated sets, and checkpoints are ignored by Git.

## Attribution

Derived from Ji Gan and Weiqiang Wang's HiGAN implementation under its MIT license.
Please cite the original *HiGAN: Handwriting Imitation Conditioned on Arbitrary-Length
Texts and Disentangled Styles* paper when using this work.
"# HiGANCon" 
