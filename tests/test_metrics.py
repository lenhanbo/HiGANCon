import torch
from torch import nn

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
