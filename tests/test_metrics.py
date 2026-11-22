import numpy as np
import torch
from torch import nn

from evaluate import (ctc_decode, higan_inception_features, kid, retrieval_metrics,
                      select_query_texts, trim_batch)
from higan_contrastive.metrics import handwriting_distance
from higan_contrastive.spisgan_eval import (spisgan_fid_kid,
                                             spisgan_inception_batch)


def test_hwd_zero_for_identical_writer_sets():
    backbone = nn.Sequential(nn.AvgPool2d(2))
    images = [torch.full((1, 32, 40), -1.0), torch.full((1, 32, 48), 1.0)]
    writers = ["a", "b"]
    assert handwriting_distance(images, images, writers, backbone, torch.device("cpu")) == 0.0


def test_hwd_detects_changed_handwriting_pixels():
    backbone = nn.Sequential(nn.AvgPool2d(2))
    real = [torch.full((1, 32, 40), -1.0), torch.full((1, 32, 40), -1.0)]
    fake = [torch.full((1, 32, 40), 1.0), torch.full((1, 32, 40), -1.0)]
    score = handwriting_distance(real, fake, ["a", "b"], backbone, torch.device("cpu"))
    assert score > 0


def test_kid_subsets_are_deterministic_and_finite():
    rng = np.random.default_rng(7)
    real = rng.normal(size=(12, 8)).astype(np.float32)
    fake = rng.normal(size=(12, 8)).astype(np.float32)
    first = kid(real, fake, subsets=4, subset_size=6, seed=42)
    second = kid(real, fake, subsets=4, subset_size=6, seed=42)
    assert np.isfinite(first)
    assert first == second


def test_writer_retrieval_chunks_cover_all_samples():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1],
                             [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    metrics = retrieval_metrics(features, labels, chunk_size=1)
    assert metrics["writer_top1"] == 1.0
    assert metrics["writer_top5"] == 1.0
    assert metrics["separation_gap"] > 0


def test_trim_batch_caps_final_evaluation_batch():
    batch = {"images": torch.arange(12).reshape(3, 4),
             "texts": ["a", "b", "c"], "unchanged": "value"}
    trimmed = trim_batch(batch, 2)
    assert trimmed["images"].shape[0] == 2
    assert trimmed["texts"] == ["a", "b"]
    assert trimmed["unchanged"] == "value"


def test_ctc_decode_ignores_padded_timesteps():
    logits = torch.full((1, 5, 5), -10.0)
    logits[0, :2, 2] = 10.0  # "A", repeated and collapsed.
    logits[0, 2:, 3] = 10.0  # Padding would incorrectly append "B".
    assert ctc_decode(logits) == ["AB"]
    assert ctc_decode(logits, torch.tensor([2])) == ["A"]


def test_query_selection_excludes_reference_and_is_deterministic():
    import random
    originals = ["one", "two"]
    pool = ["one", "two", "three"]
    first = select_query_texts(originals, pool, random.Random(7))
    second = select_query_texts(originals, pool, random.Random(7))
    assert first == second
    assert all(query != original for query, original in zip(first, originals))


def test_higan_fid_preprocessing_pads_to_original_width_and_range():
    class RecordingInception(nn.Module):
        def forward(self, images):
            self.shape = tuple(images.shape)
            self.value_range = (float(images.min()), float(images.max()))
            return [images.mean(dim=(2, 3), keepdim=True)]

    model = RecordingInception()
    images = [torch.full((1, 32, 17), -1.0), torch.full((1, 32, 41), 1.0)]
    features = higan_inception_features(
        model, images, torch.device("cpu"), max_img_width=320, batch_size=16)
    assert model.shape == (2, 3, 32, 320)
    assert model.value_range == (0.0, 1.0)
    assert features.shape == (2, 3)


def test_spisgan_fid_preprocessing_uses_fixed_white_32x128_canvas():
    class RecordingInception(nn.Module):
        def forward(self, images):
            self.images = images.detach().clone()
            return [images.mean(dim=(2, 3), keepdim=True)]

    model = RecordingInception()
    images = torch.full((1, 1, 32, 160), -1.0)
    images[:, :, :, :32] = 1.0
    features = spisgan_inception_batch(
        model, images, torch.tensor([32]), torch.device("cpu"))
    assert model.images.shape == (1, 3, 32, 128)
    assert torch.all(model.images[:, :, :, 32:] == 1.0)
    assert features.shape == (1, 3)


def test_spisgan_fid_kid_identical_features_are_zero():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(12, 8)).astype(np.float64)
    fid_value, kid_value, metadata = spisgan_fid_kid(features, features.copy())
    assert abs(fid_value) < 1e-8
    assert np.isfinite(kid_value)
    assert metadata["fid_fixed_image_size"] == [32, 128]
