from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader

from fid_kid.inception import InceptionV3
from lib.alphabet import Alphabets
from networks.module import StyleEncoder, WriterIdentifier
from networks.multiscale_generator import (build_generator, encode_reference_context,
                                           generate_with_reference)

from .data import collate_words


SPISGAN_TRAIN_WRITERS = 339
SPISGAN_TEST_WRITERS = 161
SPISGAN_FID_WIDTH = 128


def _trim_batch(batch: dict, count: int) -> dict:
    return {
        key: (value[:count] if isinstance(value, (torch.Tensor, list)) else value)
        for key, value in batch.items()
    }


def spisgan_inception_batch(model, images: torch.Tensor, lengths: torch.Tensor,
                            device: torch.device) -> np.ndarray:
    """Reproduce the preprocessing in SpiS-GAN's released ``fid_kid.py``.

    The released evaluator changes inter-sample ``-1`` padding to white, maps the
    tensor from ``[-1, 1]`` to ``[0, 1]``, repeats grayscale to RGB, then crops or
    pads every batch to a fixed 1:4 canvas (32x128).
    """
    images = images.to(device).clone()
    lengths = lengths.detach().to("cpu").long()
    batch_width = int(images.shape[-1])
    for index, length in enumerate(lengths.tolist()):
        length = max(0, min(int(length), batch_width))
        if length < batch_width:
            padding = images[index, :, :, length:]
            if bool(torch.all(padding == -1)):
                padding.fill_(1.0)
    images = ((images + 1.0) / 2.0).repeat(1, 3, 1, 1)
    target_width = 4 * int(images.shape[-2])
    if images.shape[-1] < target_width:
        images = F.pad(images, (0, target_width - images.shape[-1]), value=1.0)
    elif images.shape[-1] > target_width:
        images = images[:, :, :, :target_width]
    features = model(images)[0]
    if features.shape[-2:] != (1, 1):
        features = F.adaptive_avg_pool2d(features, (1, 1))
    return features.detach().cpu().reshape(len(images), -1).numpy()


def spisgan_fid_kid(real_features: np.ndarray, fake_features: np.ndarray
                     ) -> tuple[float, float, dict[str, object]]:
    if len(real_features) < 2 or len(fake_features) < 2:
        raise ValueError("SpiS-GAN FID/KID requires at least two real and fake samples")
    real_mean, fake_mean = real_features.mean(0), fake_features.mean(0)
    real_cov = np.cov(real_features, rowvar=False)
    fake_cov = np.cov(fake_features, rowvar=False)
    try:
        covariance_result = linalg.sqrtm(real_cov @ fake_cov, disp=False)
    except TypeError:  # SciPy >= 1.16 removed the deprecated ``disp`` argument.
        covariance_result = linalg.sqrtm(real_cov @ fake_cov)
    covariance = (covariance_result[0] if isinstance(covariance_result, tuple)
                  else covariance_result)
    if not np.isfinite(covariance).all():
        offset = np.eye(real_cov.shape[0]) * 1e-6
        try:
            covariance_result = linalg.sqrtm(
                (real_cov + offset) @ (fake_cov + offset), disp=False)
        except TypeError:
            covariance_result = linalg.sqrtm((real_cov + offset) @ (fake_cov + offset))
        covariance = (covariance_result[0] if isinstance(covariance_result, tuple)
                      else covariance_result)
    if np.iscomplexobj(covariance):
        covariance = covariance.real
    difference = real_mean - fake_mean
    fid_value = float(
        difference @ difference
        + np.trace(real_cov + fake_cov - 2.0 * covariance))
    subset_size = min(1000, len(real_features), len(fake_features))
    dimension = real_features.shape[1]
    # Match ``np.random.seed(42)`` + ``numpy.random.choice`` used upstream.
    rng = np.random.RandomState(42)
    mmd_values = []
    for _ in range(50):
        real = real_features[rng.choice(len(real_features), subset_size, replace=False)]
        fake = fake_features[rng.choice(len(fake_features), subset_size, replace=False)]
        k_rr = (real @ real.T / dimension + 1.0) ** 3
        k_ff = (fake @ fake.T / dimension + 1.0) ** 3
        k_rf = (real @ fake.T / dimension + 1.0) ** 3
        denominator = subset_size * (subset_size - 1)
        mmd_values.append(
            (k_rr.sum() - np.trace(k_rr)) / denominator
            + (k_ff.sum() - np.trace(k_ff)) / denominator
            - 2.0 * k_rf.mean())
    mmd_values = np.asarray(mmd_values)
    metadata = {
        "fid_kid_protocol": "spisgan_released_code",
        "fid_inception": "pytorch-fid InceptionV3",
        "fid_feature_dims": 2048,
        "fid_fixed_image_size": [32, SPISGAN_FID_WIDTH],
        "fid_padding_value_after_normalization": 1.0,
        "kid_subsets": 50,
        "kid_subset_size": subset_size,
        "kid_scale": 100.0,
    }
    return float(fid_value), float(mmd_values.mean() * 100.0), metadata


@torch.inference_mode()
def evaluate_spisgan_fid(args, cfg, checkpoint: dict, dataset, device: torch.device,
                         architecture_metadata: dict[str, object]) -> dict[str, object]:
    """Evaluate released-code SpiS-GAN test replication FID/KID.

    This intentionally implements the public repository's executable protocol:
    all ``test_32.hdf5`` images act as references, their original transcripts are
    regenerated, and global FID/KID are computed with the released preprocessing.
    """
    expected_classes = len(Alphabets["spisgan_iam"])
    configured_classes = int(cfg.GenModel.n_class)
    if configured_classes != expected_classes:
        raise ValueError(
            "SpiS-GAN HDF5 uses the 81-class spisgan_iam alphabet; set GenModel, "
            f"DiscModel, and OcrModel n_class to {expected_classes} before training "
            f"and evaluation (configured GenModel.n_class={configured_classes})")
    writer_count = len(dataset.writer_to_index)
    if writer_count != SPISGAN_TEST_WRITERS:
        raise ValueError(
            f"SpiS-GAN test split must contain {SPISGAN_TEST_WRITERS} writers; "
            f"got {writer_count}")

    cfg.WidModel.n_writer = checkpoint["models"]["W"]["linear_wid.2.weight"].shape[0]
    G = build_generator(cfg).to(device)
    E = StyleEncoder(**cfg.EncModel).to(device)
    W = WriterIdentifier(**cfg.WidModel).to(device)
    for name, model in (("G", G), ("E", E), ("W", W)):
        model.load_state_dict(checkpoint["models"][name])
        model.eval()

    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block]).eval().to(device)
    sample_limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    if sample_limit < 2:
        raise ValueError("SpiS-GAN evaluation requires at least two samples")
    loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
                        drop_last=False, collate_fn=collate_words, num_workers=0)
    total_batches = math.ceil(sample_limit / int(cfg.training.batch_size))
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(int(args.eval_seed))
    real_features, fake_features = [], []
    processed = 0
    started = time.perf_counter()
    print(
        f"[eval:spisgan] test replication samples={sample_limit} "
        f"writers={writer_count} batches={total_batches}", flush=True)

    for batch_index, batch in enumerate(loader, 1):
        remaining = sample_limit - processed
        if remaining <= 0:
            break
        if len(batch["texts"]) > remaining:
            batch = _trim_batch(batch, remaining)
        real = batch["images"].to(device)
        lengths = batch["image_lens"].to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)
        mu = E(real, lengths, W.cnn_backbone)
        reference_context = encode_reference_context(G, real, lengths)
        noise_shape = (len(real), int(cfg.GenModel.style_dim) - int(cfg.EncModel.style_dim))
        noise = torch.randn(noise_shape, device=device, generator=eval_generator)
        generated = generate_with_reference(
            G, torch.cat((noise, mu), 1), labels, label_lens,
            reference_context=reference_context)
        generated_lens = label_lens * int(cfg.char_width)
        real_features.append(spisgan_inception_batch(inception, real, lengths, device))
        fake_features.append(
            spisgan_inception_batch(inception, generated, generated_lens, device))
        processed += len(real)
        if (batch_index == 1 or batch_index % max(1, int(args.log_every)) == 0
                or processed >= sample_limit):
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            eta = (sample_limit - processed) / rate if rate else 0.0
            print(
                f"[eval:spisgan] batch {batch_index}/{total_batches} | "
                f"samples {processed}/{sample_limit} | {rate:.2f} samples/s | "
                f"ETA {eta:.1f}s", flush=True)

    real_array = np.concatenate(real_features, axis=0)
    fake_array = np.concatenate(fake_features, axis=0)
    gate_values_fn = getattr(G, "gate_values", None)
    gate_values = gate_values_fn() if gate_values_fn is not None else None
    del inception, G, E, W
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[eval:spisgan] computing global FID/KID", flush=True)
    fid_value, kid_value, protocol_metadata = spisgan_fid_kid(real_array, fake_array)
    return {
        "evaluation_setup": "spisgan",
        "spisgan_scope": "released-code test-set replication (global FID/KID)",
        "fid": fid_value,
        "kid": kid_value,
        "samples": int(processed),
        "test_writers": int(writer_count),
        "expected_train_writers": SPISGAN_TRAIN_WRITERS,
        "expected_test_writers": SPISGAN_TEST_WRITERS,
        "dataset_format": "spisgan_hdf5",
        "eval_seed": int(args.eval_seed),
        "generator_architecture": architecture_metadata,
        "multiscale_gate_values": gate_values,
        "iv_oov_grid_included": False,
        **protocol_metadata,
    }
