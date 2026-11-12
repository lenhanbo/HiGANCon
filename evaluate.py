from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3

from higan_contrastive.checkpoint import file_hash
from higan_contrastive.data import collate_words, load_experiment_dataset
from higan_contrastive.eval_preview import save_evaluation_previews
from higan_contrastive.metrics import (HWD_VGG16_URL, handwriting_distance,
                                       load_hwd_backbone)
from lib.alphabet import Alphabets, get_lexicon
from lib.utils import yaml2config
from networks.BigGAN_networks import Generator
from networks.module import Recognizer, StyleEncoder, WriterIdentifier


def edit_distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1]


def ctc_decode(logits: torch.Tensor) -> list[str]:
    alphabet = Alphabets["all"]
    output = []
    for sequence in logits.argmax(dim=-1).cpu().tolist():
        chars, previous = [], None
        for index in sequence:
            if index != 0 and index != previous and index < len(alphabet):
                chars.append(alphabet[index])
            previous = index
        output.append("".join(chars))
    return output


def torchvision_inception_features(model, images: list[torch.Tensor],
                                   device: torch.device) -> np.ndarray:
    """Legacy project evaluator: squash every word to 32x128 before ImageNet Inception."""
    result = []
    for start in range(0, len(images), 32):
        batch = torch.cat([
            F.interpolate(image.unsqueeze(0), (32, 128), mode="bilinear", align_corners=False)
            for image in images[start:start + 32]
        ]).to(device)
        batch = F.interpolate(batch.repeat(1, 3, 1, 1), (299, 299), mode="bilinear", align_corners=False)
        batch = (batch + 1) / 2
        mean = batch.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = batch.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        batch = (batch - mean) / std
        result.append(model(batch).cpu())
    return torch.cat(result).numpy()


def higan_inception_features(model, images: list[torch.Tensor], device: torch.device,
                             max_img_width: int, batch_size: int = 16) -> np.ndarray:
    """Match HiGAN's FID input path: pad to >=320, then use patched FID Inception.

    Images use HiGAN's [-1, 1] convention (background=-1). The upstream evaluator
    converts them to [0, 1], repeats grayscale to RGB, and lets its Inception wrapper
    resize to 299x299 and normalize to [-1, 1].
    """
    result = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        width = max(int(max_img_width), max(int(image.shape[-1]) for image in chunk))
        batch = torch.full((len(chunk), 1, 32, width), -1.0, dtype=chunk[0].dtype)
        for index, image in enumerate(chunk):
            batch[index, :, :, :image.shape[-1]] = image
        batch = ((batch.to(device) + 1) / 2).repeat(1, 3, 1, 1)
        features = model(batch)[0]
        if features.shape[-2:] != (1, 1):
            features = F.adaptive_avg_pool2d(features, (1, 1))
        result.append(features.cpu().reshape(len(chunk), -1))
    return torch.cat(result).numpy()


def fid(real: np.ndarray, fake: np.ndarray) -> float:
    mu1, mu2 = real.mean(0), fake.mean(0)
    sigma1, sigma2 = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    covariance = linalg.sqrtm(sigma1 @ sigma2, disp=False)[0]
    if np.iscomplexobj(covariance):
        covariance = covariance.real
    return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * covariance))


def kid(real: np.ndarray, fake: np.ndarray, subsets: int = 50,
        subset_size: int = 1000, seed: int = 42) -> float:
    n = min(len(real), len(fake))
    real, fake = real[:n], fake[:n]
    if n < 2:
        raise ValueError("KID requires at least two samples")
    dimension = real.shape[1]
    subset_size = min(int(subset_size), n)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(int(subsets)):
        real_subset = real[rng.choice(n, subset_size, replace=False)]
        fake_subset = fake[rng.choice(n, subset_size, replace=False)]
        k_rr = (real_subset @ real_subset.T / dimension + 1) ** 3
        k_ff = (fake_subset @ fake_subset.T / dimension + 1) ** 3
        k_rf = (real_subset @ fake_subset.T / dimension + 1) ** 3
        score = ((k_rr.sum() - np.trace(k_rr)) / (subset_size * (subset_size - 1))
                 + (k_ff.sum() - np.trace(k_ff)) / (subset_size * (subset_size - 1))
                 - 2 * k_rf.mean())
        scores.append(score)
    return float(np.mean(scores))


def fid_kid_for_protocol(protocol: str, real_images: list[torch.Tensor],
                         fake_images: list[torch.Tensor], device: torch.device,
                         max_img_width: int) -> tuple[float, float, dict[str, object]]:
    if protocol == "higan":
        from fid_kid.fid_kid_pad import (calculate_frechet_distance,
                                         polynomial_mmd_averages)
        from fid_kid.inception import InceptionV3

        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block]).eval().to(device)
        real_features = higan_inception_features(
            model, real_images, device, max_img_width=max_img_width)
        fake_features = higan_inception_features(
            model, fake_images, device, max_img_width=max_img_width)
        del model
        real_mean, fake_mean = real_features.mean(0), fake_features.mean(0)
        real_cov = np.cov(real_features, rowvar=False)
        fake_cov = np.cov(fake_features, rowvar=False)
        fid_value = calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
        # Call the upstream implementation itself; HiGAN reports its mean MMD x100.
        numpy_rng = np.random.get_state()
        try:
            np.random.seed(42)
            mmd_values, _ = polynomial_mmd_averages(
                real_features, fake_features, degree=3, gamma=None, coef0=1.0,
                ret_var=True, n_subsets=50, subset_size=1000)
        finally:
            np.random.set_state(numpy_rng)
        kid_value = float(mmd_values.mean() * 100.0)
        metadata = {
            "fid_kid_protocol": "higan",
            "fid_inception": "tensorflow-fid patched InceptionV3",
            "fid_feature_dims": 2048,
            "fid_min_image_width": int(max_img_width),
            "kid_subsets": 50,
            "kid_subset_size": min(1000, len(real_features), len(fake_features)),
            "kid_scale": 100.0,
        }
        return float(fid_value), float(kid_value), metadata

    if protocol == "torchvision":
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        real_features = torchvision_inception_features(model, real_images, device)
        fake_features = torchvision_inception_features(model, fake_images, device)
        del model
        metadata = {
            "fid_kid_protocol": "torchvision",
            "fid_inception": "torchvision ImageNet InceptionV3",
            "fid_feature_dims": 2048,
            "fid_fixed_image_size": [32, 128],
            "kid_subsets": 50,
            "kid_subset_size": min(1000, len(real_features), len(fake_features)),
            "kid_scale": 1.0,
        }
        return fid(real_features, fake_features), kid(real_features, fake_features), metadata

    raise ValueError(f"Unknown FID/KID protocol: {protocol}")


def retrieval_metrics(features: torch.Tensor, labels: torch.Tensor,
                      chunk_size: int = 256) -> dict[str, float]:
    features = F.normalize(features, dim=1)
    n = len(labels)
    if n < 2:
        raise ValueError("Writer retrieval requires at least two samples")
    top_k = min(5, n - 1)
    top1_hits = top5_hits = 0
    positive_sum = negative_sum = 0.0
    positive_count = negative_count = 0
    all_indices = torch.arange(n)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        similarity = features[start:end] @ features.T
        local_indices = torch.arange(end - start)
        similarity[local_indices, all_indices[start:end]] = -float("inf")
        ranking = similarity.topk(top_k, dim=1).indices
        matches = labels[ranking].eq(labels[start:end, None])
        top1_hits += int(matches[:, :1].any(1).sum())
        top5_hits += int(matches.any(1).sum())

        raw_similarity = features[start:end] @ features.T
        same = labels[start:end, None].eq(labels[None, :])
        same[local_indices, all_indices[start:end]] = False
        different = ~labels[start:end, None].eq(labels[None, :])
        positive_sum += float(raw_similarity[same].sum())
        negative_sum += float(raw_similarity[different].sum())
        positive_count += int(same.sum())
        negative_count += int(different.sum())
    pos = positive_sum / max(1, positive_count)
    neg = negative_sum / max(1, negative_count)
    return {"writer_top1": top1_hits / n, "writer_top5": top5_hits / n,
            "positive_cosine": pos, "negative_cosine": neg, "separation_gap": pos - neg}


def trim_batch(batch: dict, count: int) -> dict:
    trimmed = {}
    for key, value in batch.items():
        trimmed[key] = value[:count] if isinstance(value, (torch.Tensor, list)) else value
    return trimmed


def encode_query_texts(texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    alphabet = {char: index for index, char in enumerate(Alphabets["all"])}
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.int32, device=device)
    labels = torch.zeros((len(texts), int(lengths.max())), dtype=torch.long, device=device)
    for row, text in enumerate(texts):
        labels[row, :len(text)] = torch.tensor(
            [alphabet[char] for char in text], dtype=torch.long, device=device)
    return labels, lengths


@torch.inference_mode()
def evaluate(args) -> dict:
    cfg = yaml2config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = json.loads((Path(args.manifest_root) / "split_metadata.json").read_text(encoding="utf-8"))
    metadata_format = metadata.get("dataset_format", "iam_raw")
    if metadata_format != args.dataset_format:
        raise ValueError(
            f"Dataset format mismatch: CLI={args.dataset_format}, metadata={metadata_format}")
    if checkpoint["config_hash"] != file_hash(args.config):
        raise ValueError("Checkpoint/config hash mismatch")
    if checkpoint["split_hash"] != metadata["split_hash"]:
        raise ValueError("Checkpoint/manifest split hash mismatch")
    if checkpoint.get("dataset_format", "iam_raw") != args.dataset_format:
        raise ValueError("Checkpoint dataset format does not match evaluation dataset")
    dataset = load_experiment_dataset(
        args.dataset_format, args.dataset_root, args.manifest_root, "test")
    cfg.WidModel.n_writer = checkpoint["models"]["W"]["linear_wid.2.weight"].shape[0]
    G = Generator(**cfg.GenModel).to(device)
    E = StyleEncoder(**cfg.EncModel).to(device)
    W = WriterIdentifier(**cfg.WidModel).to(device)
    R = Recognizer(**cfg.OcrModel).to(device)
    for name, model in (("G", G), ("E", E), ("W", W), ("R", R)):
        model.load_state_dict(checkpoint["models"][name])
        model.eval()
    independent = bool(args.recognizer_checkpoint)
    if independent:
        recognizer_state = torch.load(args.recognizer_checkpoint, map_location=device, weights_only=False)
        R.load_state_dict(recognizer_state.get("R", recognizer_state.get("Recognizer", recognizer_state)))

    loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
                        drop_last=False, collate_fn=collate_words, num_workers=0)
    real_images, fake_images, hwd_real, hwd_fake, hwd_writers = [], [], [], [], []
    embeddings, writer_labels = [], []
    preview_samples = []
    preview_rng = random.Random(int(args.eval_seed) + 1_000_003)
    preview_lexicon = []
    if args.preview_query_mode == "random" and not args.preview_query_text:
        preview_lexicon = get_lexicon(
            cfg.training.lexicon, Alphabets["all"], max_length=int(cfg.training.max_word_len))
        if not preview_lexicon:
            raise ValueError("Preview random-query lexicon is empty")
    char_errors = word_errors = char_total = word_total = 0
    diversities = []
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(int(args.eval_seed))
    for batch in loader:
        if args.max_samples > 0:
            remaining = args.max_samples - len(real_images)
            if remaining <= 0:
                break
            if len(batch["texts"]) > remaining:
                batch = trim_batch(batch, remaining)
        real = batch["images"].to(device)
        lengths = batch["image_lens"].to(device)
        metric_real = batch.get("metric_real_images", batch["images"]).to(device)
        metric_real_lens = batch.get("metric_real_lens", batch["image_lens"]).to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)
        mu = E(real, lengths, W.cnn_backbone)
        noise_shape = (len(real), int(cfg.GenModel.style_dim) - int(cfg.EncModel.style_dim))
        noise1 = torch.randn(noise_shape, device=device, generator=eval_generator)
        noise2 = torch.randn(noise_shape, device=device, generator=eval_generator)
        generated = G(torch.cat((noise1, mu), 1), labels, label_lens)
        generated2 = G(torch.cat((noise2, mu), 1), labels, label_lens)
        generated_lens = label_lens * int(cfg.char_width)
        if args.preview_samples < 0:
            preview_count = len(real)
        else:
            preview_count = min(len(real), max(0, args.preview_samples - len(preview_samples)))
        gallery_queries = list(batch["texts"][:preview_count])
        gallery_generated = generated[:preview_count]
        gallery_lens = generated_lens[:preview_count]
        if preview_count and (args.preview_query_text or args.preview_query_mode == "random"):
            if args.preview_query_text:
                gallery_queries = [args.preview_query_text] * preview_count
            else:
                gallery_queries = []
                for original in batch["texts"][:preview_count]:
                    candidates = preview_lexicon
                    query = preview_rng.choice(candidates)
                    if len(candidates) > 1:
                        while query == original:
                            query = preview_rng.choice(candidates)
                    gallery_queries.append(query)
            gallery_labels, gallery_label_lens = encode_query_texts(gallery_queries, device)
            gallery_generated = G(
                torch.cat((noise1[:preview_count], mu[:preview_count]), 1),
                gallery_labels, gallery_label_lens)
            gallery_lens = gallery_label_lens * int(cfg.char_width)
        embeddings.append(E(generated, generated_lens, W.cnn_backbone).cpu())
        writer_labels.append(batch["writers"])
        diversities.append((generated - generated2).abs().mean().item())
        predictions = ctc_decode(R(generated))
        for truth, prediction in zip(batch["texts"], predictions):
            char_errors += edit_distance(truth, prediction)
            char_total += len(truth)
            word_errors += edit_distance(truth.split(), prediction.split())
            word_total += len(truth.split())
        for index in range(len(real)):
            reference_image = real[index, :, :, :int(lengths[index])].cpu()
            real_image = metric_real[index, :, :, :int(metric_real_lens[index])].cpu()
            fake_image = generated[index, :, :, :int(generated_lens[index])].cpu()
            sample_index = len(real_images)
            real_images.append(real_image)
            fake_images.append(fake_image)
            if index < preview_count:
                gallery_image = gallery_generated[index, :, :, :int(gallery_lens[index])].cpu()
                preview_samples.append({
                    "sample_index": sample_index,
                    "writer_id": batch["writer_ids"][index],
                    "word_id": batch["word_ids"][index],
                    "reference_text": batch["texts"][index],
                    "query": gallery_queries[index],
                    "reference": reference_image,
                    "generated": gallery_image,
                })
            if not args.skip_hwd:
                hwd_real.append(real_image)
                hwd_fake.append(fake_image)
                hwd_writers.append(batch["writer_ids"][index])
    del G, E, W, R, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    max_img_width = int(cfg.char_width) * int(cfg.training.max_word_len)
    fid_value, kid_value, protocol_metadata = fid_kid_for_protocol(
        args.fid_kid_protocol, real_images, fake_images, device, max_img_width)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = {"fid": fid_value, "kid": kid_value,
               "cer": char_errors / max(1, char_total), "wer": word_errors / max(1, word_total),
               "diversity_l1": float(np.mean(diversities)),
               "independent_recognizer": independent, "samples": len(real_images),
               "eval_seed": int(args.eval_seed), "dataset_format": args.dataset_format,
               **protocol_metadata}
    if args.skip_hwd:
        metrics.update({"hwd": None, "hwd_skipped": True})
    else:
        hwd_backbone = load_hwd_backbone(device, args.hwd_checkpoint or None)
        hwd_value = handwriting_distance(hwd_real, hwd_fake, hwd_writers, hwd_backbone, device)
        del hwd_backbone
        metrics.update({"hwd": hwd_value, "hwd_skipped": False,
                        "hwd_writers": len(set(hwd_writers)),
                        "hwd_backbone": args.hwd_checkpoint or HWD_VGG16_URL})
    metrics.update(retrieval_metrics(torch.cat(embeddings), torch.cat(writer_labels)))
    preview_pages = []
    if preview_samples:
        preview_root = (Path(args.preview_dir) if args.preview_dir else
                        Path(args.output).parent / "evaluation_previews")
        preview_pages = save_evaluation_previews(
            preview_samples, preview_root, page_size=int(args.preview_page_size))
    metrics.update({"preview_samples": len(preview_samples),
                    "preview_pages": [str(path) for path in preview_pages],
                    "preview_query_mode": ("fixed" if args.preview_query_text else
                                           args.preview_query_mode),
                    "preview_query_text": args.preview_query_text})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a HiGAN checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-format", choices=("iam_raw", "higan_hdf5"), default="iam_raw")
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recognizer-checkpoint", default=None,
                        help="Independent IAM recognizer state dict for reportable CER/WER")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Maximum unique test images; 0 evaluates the full manifest")
    parser.add_argument("--fid-kid-protocol", choices=("higan", "torchvision"), default="higan",
                        help="higan reproduces upstream FID/KID preprocessing, weights, and KID x100")
    parser.add_argument("--eval-seed", type=int, default=123456,
                        help="Fixed seed for generated evaluation noise")
    parser.add_argument("--preview-samples", type=int, default=0,
                        help="Gallery samples: 0 disables, positive caps, -1 saves every evaluated sample")
    parser.add_argument("--preview-page-size", type=int, default=16)
    parser.add_argument("--preview-dir", default=None,
                        help="Gallery output; defaults beside evaluation.json")
    parser.add_argument("--preview-query-mode", choices=("dataset", "random"), default="dataset",
                        help="dataset reuses each reference text; random draws a different lexicon word")
    parser.add_argument("--preview-query-text", default=None,
                        help="Fixed gallery query for every reference; overrides preview-query-mode")
    parser.add_argument("--hwd-checkpoint", default=None,
                        help="Local path or hf://owner/repo/file@revision; official GitHub download otherwise")
    parser.add_argument("--skip-hwd", action="store_true",
                        help="Skip HWD backbone download/computation; writes hwd=null")
    args = parser.parse_args()
    if args.preview_samples < -1:
        parser.error("--preview-samples must be -1, 0, or a positive integer")
    if args.preview_page_size < 1:
        parser.error("--preview-page-size must be positive")
    if args.preview_query_text:
        if len(args.preview_query_text) > 20:
            parser.error("--preview-query-text exceeds the configured 20-character maximum")
        invalid = sorted(set(args.preview_query_text) - set(Alphabets["all"]))
        if invalid:
            parser.error(f"--preview-query-text contains unsupported characters: {invalid}")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
