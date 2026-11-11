from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3

from higan_contrastive.checkpoint import file_hash
from higan_contrastive.data import IAMManifestDataset, collate_words
from higan_contrastive.metrics import (HWD_VGG16_URL, handwriting_distance,
                                       load_hwd_backbone)
from lib.alphabet import Alphabets
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


def inception_features(model, images: list[torch.Tensor], device: torch.device) -> np.ndarray:
    result = []
    for start in range(0, len(images), 32):
        batch = torch.stack(images[start:start + 32]).to(device)
        batch = F.interpolate(batch.repeat(1, 3, 1, 1), (299, 299), mode="bilinear", align_corners=False)
        batch = (batch + 1) / 2
        mean = batch.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = batch.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        batch = (batch - mean) / std
        result.append(model(batch).cpu())
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


@torch.inference_mode()
def evaluate(args) -> dict:
    cfg = yaml2config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = json.loads((Path(args.manifest_root) / "split_metadata.json").read_text(encoding="utf-8"))
    if checkpoint["config_hash"] != file_hash(args.config):
        raise ValueError("Checkpoint/config hash mismatch")
    if checkpoint["split_hash"] != metadata["split_hash"]:
        raise ValueError("Checkpoint/manifest split hash mismatch")
    dataset = IAMManifestDataset(args.dataset_root, Path(args.manifest_root) / "test.jsonl")
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
    char_errors = word_errors = char_total = word_total = 0
    diversities = []
    for batch in loader:
        if args.max_samples > 0:
            remaining = args.max_samples - len(real_images)
            if remaining <= 0:
                break
            if len(batch["texts"]) > remaining:
                batch = trim_batch(batch, remaining)
        real = batch["images"].to(device)
        lengths = batch["image_lens"].to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)
        mu = E(real, lengths, W.cnn_backbone)
        noise1 = torch.randn(len(real), int(cfg.GenModel.style_dim) - int(cfg.EncModel.style_dim), device=device)
        noise2 = torch.randn_like(noise1)
        generated = G(torch.cat((noise1, mu), 1), labels, label_lens)
        generated2 = G(torch.cat((noise2, mu), 1), labels, label_lens)
        generated_lens = label_lens * int(cfg.char_width)
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
            real_images.append(F.interpolate(real[index:index + 1], (32, 128)).squeeze(0).cpu())
            fake_images.append(F.interpolate(generated[index:index + 1], (32, 128)).squeeze(0).cpu())
            if not args.skip_hwd:
                hwd_real.append(real[index, :, :, :int(lengths[index])].cpu())
                hwd_fake.append(generated[index, :, :, :int(generated_lens[index])].cpu())
                hwd_writers.append(batch["writer_ids"][index])
    del G, E, W, R, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception.fc = torch.nn.Identity()
    inception.eval().to(device)
    real_features = inception_features(inception, real_images, device)
    fake_features = inception_features(inception, fake_images, device)
    del inception
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = {"fid": fid(real_features, fake_features), "kid": kid(real_features, fake_features),
               "cer": char_errors / max(1, char_total), "wer": word_errors / max(1, word_total),
               "diversity_l1": float(np.mean(diversities)),
               "independent_recognizer": independent, "samples": len(real_images)}
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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a HiGAN checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recognizer-checkpoint", default=None,
                        help="Independent IAM recognizer state dict for reportable CER/WER")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Maximum unique test images; 0 evaluates the full manifest")
    parser.add_argument("--hwd-checkpoint", default=None,
                        help="Local path or hf://owner/repo/file@revision; official GitHub download otherwise")
    parser.add_argument("--skip-hwd", action="store_true",
                        help="Skip HWD backbone download/computation; writes hwd=null")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
