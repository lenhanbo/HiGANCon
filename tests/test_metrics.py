import numpy as np
import torch
from torch import nn

from evaluate import higan_inception_features, kid, retrieval_metrics, trim_batch
from higan_contrastive.metrics import handwriting_distance


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
