from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Hashable, Sequence

import torch
from torch import nn
from torchvision.models import vgg16


HWD_VGG16_URL = (
    "https://github.com/aimagelab/font_square/releases/download/"
    "VGG-16/VGG16_class_10400.pth"
)


def load_hwd_backbone(device: torch.device, checkpoint: str | None = None) -> nn.Module:
    """Load the official Font² VGG-16 and retain its convolutional features."""
    model = vgg16(weights=None, num_classes=10400)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        state = torch.hub.load_state_dict_from_url(
            HWD_VGG16_URL, progress=True, map_location="cpu")
    model.load_state_dict(state)
    backbone = model.features
    backbone.eval().requires_grad_(False)
    return backbone.to(device)


def _hwd_input(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert internal white-on-black HiGAN pixels to official HWD RGB input."""
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("HWD expects a single grayscale image shaped [1,H,W]")
    # HiGAN: background=-1, ink=+1. HWD: white background=1, dark ink=0.
    image = 1.0 - (image.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0
    if image.shape[-2] != 32:
        width = max(1, round(image.shape[-1] * 32 / image.shape[-2]))
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), (32, width), mode="bilinear", align_corners=False).squeeze(0)
    if image.shape[-1] < 32:
        image = torch.nn.functional.pad(image, (0, 32 - image.shape[-1]), value=1.0)
    return image.repeat(3, 1, 1).unsqueeze(0).to(device)


@torch.inference_mode()
def _writer_feature_means(images: Sequence[torch.Tensor], writers: Sequence[Hashable],
                          backbone: nn.Module, device: torch.device) -> dict[Hashable, torch.Tensor]:
    if len(images) != len(writers) or not images:
        raise ValueError("HWD images/writers must be non-empty and have equal length")
    sums: dict[Hashable, torch.Tensor] = {}
    counts: dict[Hashable, int] = defaultdict(int)
    for image, writer in zip(images, writers):
        features = backbone(_hwd_input(image, device))
        # Official HWD treats each horizontal VGG feature location as a style feature.
        tokens = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).double().cpu()
        feature_sum = tokens.sum(dim=0)
        sums[writer] = feature_sum if writer not in sums else sums[writer] + feature_sum
        counts[writer] += len(tokens)
    return {writer: feature_sum / counts[writer] for writer, feature_sum in sums.items()}


@torch.inference_mode()
def handwriting_distance(real_images: Sequence[torch.Tensor], fake_images: Sequence[torch.Tensor],
                         writers: Sequence[Hashable], backbone: nn.Module,
                         device: torch.device) -> float:
    """Official writer-conditioned HWD: mean Euclidean distance across writers."""
    if len(real_images) != len(fake_images):
        raise ValueError("HWD real/fake sample counts differ")
    real_means = _writer_feature_means(real_images, writers, backbone, device)
    fake_means = _writer_feature_means(fake_images, writers, backbone, device)
    if real_means.keys() != fake_means.keys():
        raise ValueError("HWD real/fake writer sets differ")
    distances = [torch.linalg.vector_norm(real_means[writer] - fake_means[writer])
                 for writer in sorted(real_means, key=str)]
    return torch.stack(distances).mean().item()
