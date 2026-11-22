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
lambda_con * SupCon(L2Norm(E_mu(G([shared_noise, E_mu(reference)], configurable_text))))
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

### Original HiGAN HDF5 backend

For protocol-level comparison with upstream HiGAN, set `DATASET_FORMAT=higan_hdf5`
and point `DATASET_ROOT` to a directory containing exactly:

```text
trnvalset_words32.hdf5
testset_words32.hdf5
testset_words32_OrgSz.hdf5
```

The original README sometimes names the last file
`testset_words32d_OrgSz.hdf5`; both spellings are accepted automatically.

Validate the files and create a content-derived split hash with:

```bash
python prepare_higan_hdf5.py \
  --hdf5-root "$DATASET_ROOT" \
  --output /kaggle/working/data_higan_hdf5
```

This backend follows the upstream split exactly: `trnvalset_words32.hdf5` is the
entire training set and `testset_words32.hdf5` is the style-reference test set. There
is no separate validation split. During evaluation, aligned real images come from
`testset_words32_OrgSz.hdf5`, matching HiGAN's FID/KID path. HDF5 is read lazily and
`wids` drives the same two-distinct-indices-per-writer SupCon sampler; no PNG export is
performed. The original HiGAN code evaluates on `test` from epoch 30 and chooses its
`best.pth` by test KID. Prefer the final `last.pt` when you want to avoid that test-set
model-selection behavior.

Training with this backend uses:

```bash
bash scripts/kaggle_setup.sh "$DATASET_ROOT" higan_hdf5 /kaggle/working/data_higan_hdf5
bash scripts/kaggle_run.sh \
  --dataset-root "$DATASET_ROOT" \
  --dataset-format higan_hdf5 \
  --manifest-root /kaggle/working/data_higan_hdf5 \
  --run-mode contrastive
```

On Kaggle, the helper scripts take ordinary positional arguments; they do not require
environment variables:

```bash
bash scripts/kaggle_setup.sh "$DATASET_ROOT"
bash scripts/kaggle_run.sh \
  --dataset-root "$DATASET_ROOT" \
  --run-mode "$RUN_MODE" \
  --supcon-per-item-text \
  --max-hours "$MAX_HOURS" \
  --max-steps "$MAX_STEPS" \
  --eval-max-samples "$EVAL_MAX_SAMPLES" \
  --skip-hwd
```

`RUN_MODE` accepts `baseline`, `supcon` (or legacy `contrastive`), `ctc`, and `full`.
Choose exactly one SupCon text flag: `--supcon-shared-text` or
`--supcon-per-item-text`. The Kaggle notebook exposes the same choices as
`RUN_MODE` and `SUPCON_TEXT_MODE` in its first configuration cell. It also exposes
`GENERATOR_VARIANT` (`baseline` or `multiscale_condition`) and writes that selection
back into the single `configs/config.yaml` before smoke testing.

Images are grayscale, resized to height 32 with aspect ratio preserved, inverted and
normalized to `[-1,1]`, then padded with `-1` to a width divisible by 16.

## Training and resume

```bash
python train.py \
  --config configs/config.yaml \
  --dataset-root "$DATASET_ROOT" \
  --manifest-root /kaggle/working/data \
  --output /kaggle/working/outputs/contrastive \
  --max-hours 11.5
```

`configs/config.yaml` is the single configuration for HiGAN, SupCon, CTC-guided
matching, schedules, previews, automatic evaluation, and every model. Use loss opt-out flags to select an
ablation while keeping the same configuration.

Generator architecture is also YAML-owned. Keep the exact original training/model path with:

```yaml
architecture:
  generator_variant: baseline
```

To enable the opt-in hierarchical reference encoder, change only that value to
`multiscale_condition`. It extracts padding-aware style descriptors at image heights
32, 16, and 8, adds them through learnable gates to HiGAN's three existing 32-D style
chunks, and then uses the original conditional BatchNorm and upsampling blocks. Random
`G(z, text)` generation bypasses this encoder. The global 96-D `StyleEncoder`, all loss
modes, schedules, optimizer steps, SupCon text flags, previews, and automatic evaluation
remain unchanged. Use a separate output directory for each architecture; the Kaggle script
automatically appends `-multiscale_condition` to non-baseline output names.

Resume with the exact same config and manifests:

```bash
python train.py ... --resume /kaggle/input/previous-output/outputs/contrastive/checkpoints/last.pt
```

Checkpoints include `G/D/E/R/W`, generator architecture and learned multiscale gates,
optimizers, schedulers, AMP scaler, epoch/global step,
Python/NumPy/PyTorch/CUDA RNG, sampler position, Git commit, config/split hashes, and
initial-state hash. `--max-hours` saves `last.pt` before stopping.

`--max-steps N` is an exact global-step ceiling. It saves `last.pt` immediately after
step N; `0` disables this ceiling. On resume the restored global step is used, so raising
the ceiling continues training while lowering it below the saved step exits immediately.

Kaggle runs save a deterministic visual panel every 500 steps under `previews/` (the
YAML default remains 200 for checkpoint compatibility). It shows
`REAL REFERENCE | RANDOM SYNTHETIC STYLE | REFERENCE-GUIDED | SUPCON PER-ITEM TEXT/SHARED NOISE`.
The random column uses one `z ~ N(0,I)` shared by every row, so multiple texts reveal one
synthetic style; it has no real writer identity. The adjacent JSON records its seed and
all 128 latent values. SupCon rows preserve sampler order, so each adjacent pair belongs
to one writer and uses the same text/noise. Preview generation uses a local RNG and eval
mode, so it does not advance training RNG or BatchNorm statistics. Change
Pass `--preview-every-steps N` to `train.py` or `scripts/kaggle_run.sh` to change the
runtime interval without changing the config hash; `0` disables training previews.

SupCon text mode defaults to `contrastive.text_mode: per_item`, so every item keeps its
own query text. Use `--supcon-shared-text` to run the original shared-text variant, or
`--supcon-per-item-text` to explicitly select per-item text. These two flags are
mutually exclusive and override the YAML value for that run.

Console loss records are averages over `print_every_steps` training batches. This is
important because D-side runs every batch while G/E-side runs every four batches; the
windowed record still includes `g_total`, `supcon`, and `lambda_con`. Per-batch values
remain in `metrics.jsonl`. `lambda_con` and `supcon` are zero before
`contrastive.start_step`; the weight then rises per training step over
`contrastive.ramp_steps`. Set the YAML ramp to 0 or 1 to switch immediately to the
target weight.

If batch 16 is too large, change `writers_per_batch` to 4 and `batch_size` to 8 in
**both** configs before starting either experiment. This gives 4 writers × 2 references.

### CTC-guided soft matching

The unified configuration keeps every auxiliary schedule in `configs/config.yaml`:

```bash
python train.py \
  --config configs/config.yaml \
  --dataset-root "$DATASET_ROOT" \
  --manifest-root data/manifests \
  --output outputs/ctc-soft
```

SupCon and CTC-guided matching are enabled by default. Disable either with
`--no supcon` / `--no ctc-guided` (or `--no-supcon` / `--no-ctc-guided`), and repeat
`--no` to disable both. In CTC-soft mode, `train.py` intentionally has no CLI overrides for loss
schedules or preview cadence. Its previews include the paired target and generated
word, optional SupCon view, CTC heatmaps, and per-sample alignment JSON.

After every completed training run, `train.py` automatically evaluates
`checkpoints/last.pt` on the test split and writes `evaluation.json` under the run
output directory. This also happens after `--max-steps` or `--max-hours`; smoke tests
skip it. Use `--no-eval` to opt out, `--eval-max-samples N` for a limited evaluation,
or `--eval-skip-hwd` when the HWD backbone is unavailable.

## Evaluation

```bash
python evaluate.py \
  --config configs/config.yaml \
  --dataset-root "$DATASET_ROOT" \
  --manifest-root /kaggle/working/data \
  --checkpoint /kaggle/working/outputs/contrastive/checkpoints/last.pt \
  --recognizer-checkpoint /kaggle/input/independent-iam-recognizer/model.pt \
  --fid-kid-protocol higan \
  --eval-seed 123456 \
  --output /kaggle/working/outputs/contrastive/evaluation.json
```

The report contains FID, KID, HWD, writer retrieval Top-1/Top-5, positive/negative
cosine, separation gap, noise diversity, and length-aware CER/WER for four views:
recognizer calibration on real test images, same-text generation, different in-vocabulary
text generation, and OOV-text generation. The legacy `cer`/`wer` keys alias the
same-text values. Different-text queries come from IAM training transcripts; OOV queries
come from the configured open lexicon after removing training transcripts. HWD follows the official
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
When this value is false, CER/WER are diagnostic only. For reportable text metrics,
provide one recognizer checkpoint trained only on real data and use that same frozen
checkpoint for every generator being compared. The report includes
`recognizer_real_cer` and `recognizer_real_wer` so its baseline quality is visible.

`--fid-kid-protocol higan` is the Kaggle default and reproduces the upstream HiGAN
metric settings: images are padded to at least `char_width * max_word_len = 320`,
features come from the patched TensorFlow-FID Inception at dimension 2048, and KID
uses a degree-3 polynomial kernel over 50 subsets of at most 1,000 samples. As in
HiGAN, reported KID is multiplied by 100. The output records the protocol, backbone,
subset settings, width and KID scale. `--fid-kid-protocol torchvision` retains this
repository's previous fixed-`32x128` ImageNet-Inception evaluator for old-run
comparisons. `--eval-seed 123456` fixes generation noise so baseline and contrastive
receive identical evaluation noise.

Evaluation walks `test.jsonl` sequentially with no shuffle and no dropped final batch,
so every test image is evaluated exactly once. `EVAL_MAX_SAMPLES=0` (the default) means
the complete test manifest; set a positive value only for a quicker capped run. KID is
estimated over 50 deterministic subsets of at most 1,000 samples, and writer retrieval
is computed in chunks, allowing the complete test split to be evaluated without
materializing their full pairwise matrices.

Success means improved writer retrieval and separation gap, CER degradation at most
1 percentage point, and KID degradation at most 5% versus baseline.

### SpiS-GAN test-replication FID/KID

The opt-in SpiS-GAN evaluator follows the executable protocol in the authors' released
code: use every image in the official `test_32.hdf5` as a style reference, regenerate its
own transcript, and compute global FID/KID with patched Inception-2048 on a fixed 32x128
white-padded/cropped canvas. It does not claim to reproduce the paper-only IV/OOV grid or
writer-mean FID because the public repository does not provide their sampling runner.

Download and validate the official 339/161-writer split:

```bash
python scripts/download_hf_hdf5.py DAIR-Group/SpiS-GAN \
  --repo-type model --prefix data --dataset-format spisgan_hdf5 \
  --output-dir data/spisgan_hdf5
python prepare_higan_hdf5.py \
  --hdf5-root data/spisgan_hdf5 \
  --dataset-format spisgan_hdf5 \
  --output data/spisgan_hdf5_metadata
```

The released files currently validate as follows:

- `train_32.hdf5`: 66,603 images, 339 writers, SHA-256
  `D31090DC037554619FDB70D53AA90597F3A69D0CB8EB1DFD833A44E99395B0D0`
- `test_32.hdf5`: 25,766 images, 161 writers, SHA-256
  `5E832EC551A57A27D6824BE57F1A01CE5CBF10172E4C803F8E3AA5A299B63790`

This release uses the 81-class `spisgan_iam` alphabet. Before training on it, set all
three `n_class` values (`GenModel`, `DiscModel`, and `OcrModel`) in
`configs/config.yaml` to `81`. A checkpoint trained on the HiGAN 72-class, 372/128 split
is deliberately rejected because it is not a valid unseen-writer SpiS-GAN experiment.

Evaluate a checkpoint that was trained on the same SpiS-GAN split:

```bash
python evaluate.py \
  --config configs/config.yaml \
  --dataset-root data/spisgan_hdf5 \
  --dataset-format spisgan_hdf5 \
  --manifest-root data/spisgan_hdf5_metadata \
  --checkpoint outputs/spisgan/checkpoints/last.pt \
  --eval-setup spisgan \
  --max-samples 0 \
  --output outputs/spisgan/evaluation.json
```

`--max-samples 0` evaluates all 25,766 test images. Positive values are useful only for
pipeline checks and should not be reported as the full SpiS-GAN FID.

### Evaluate a checkpoint from Hugging Face

The eval-only helper accepts a full Hub URL, `hf://` URI, or model repository ID. It
does not train. It downloads the checkpoint, creates manifests when absent, evaluates,
and saves paged `REFERENCE | QUERY TEXT | GENERATED` galleries plus JSONL metadata:

```bash
bash scripts/kaggle_eval_hf.sh \
  --checkpoint-link "https://huggingface.co/owner/repo/resolve/main/latest.pt" \
  --dataset-format higan_hdf5 \
  --run-mode contrastive \
  --eval-max-samples 0 \
  --preview-samples 64 \
  --preview-query-mode random \
  --fid-kid-protocol higan \
  --eval-seed 123456
```

`--preview-samples -1` writes a row for every evaluated test sample, `0` disables the
gallery, and a positive value writes the first N sequential samples. Pages and
`samples.jsonl` are stored under
`/kaggle/working/hf_evaluations/<mode>/evaluation_previews/`. When passing only
`owner/repo`, add `--checkpoint-filename latest.pt`; `--hf-revision` defaults to
`main`. A private repository uses the standard `HF_TOKEN` environment variable, which
can be supplied through Kaggle Secrets. The helper skips HWD by default; pass
`--with-hwd` to enable it. The helper's gallery defaults to a deterministic different
lexicon query for each reference (`--preview-query-mode random`), while metric images
continue using ground-truth queries. Use `--preview-query-mode dataset` to show the
reference's own word, or `--preview-query-text "hello"` to apply one fixed query to all
references.

When the same Hub repository contains the three root-level HDF5 files, the helper
downloads them automatically; `--dataset-root` is unnecessary. If they are stored in
a folder such as `data/`, pass `--hdf5-prefix data`. Supplying `--dataset-root` uses
already attached/downloaded HDF5 files instead.

## Kaggle and tests

Open [`kaggle/higancon.ipynb`](kaggle/higancon.ipynb), set its variables,
and run all cells. It clones a pinned ref, validates CUDA and the explicit dataset root,
installs packages without replacing Kaggle's PyTorch, creates manifests, runs tests and a
one-batch smoke update, then trains/resumes and evaluates. Outputs remain directly under
`/kaggle/working/outputs/` without creating an additional tar archive.

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
