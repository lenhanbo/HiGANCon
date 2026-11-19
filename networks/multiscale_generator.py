from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .BigGAN_networks import Generator
from .utils import _len2mask


BASELINE_VARIANT = "baseline"
MULTISCALE_VARIANT = "multiscale_condition"
SUPPORTED_GENERATOR_VARIANTS = {BASELINE_VARIANT, MULTISCALE_VARIANT}
SCALES = (8, 16, 32)


def _config_dict(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected a mapping-like config, got {type(value).__name__}")


def generator_variant(cfg) -> str:
    architecture = getattr(cfg, "architecture", None)
    variant = (getattr(architecture, "generator_variant", BASELINE_VARIANT)
               if architecture is not None else BASELINE_VARIANT)
    variant = str(variant).strip().lower()
    if variant not in SUPPORTED_GENERATOR_VARIANTS:
        choices = ", ".join(sorted(SUPPORTED_GENERATOR_VARIANTS))
        raise ValueError(f"architecture.generator_variant must be one of: {choices}")
    return variant


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualEncoderBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class HierarchicalMultiScaleEncoder(nn.Module):
    """Encode a padded 32-pixel-high style reference at H=32, 16, and 8."""

    def __init__(self, in_channels: int = 1,
                 encoder_channels: Mapping[int | str, int] | None = None):
        super().__init__()
        raw_channels = encoder_channels or {32: 64, 16: 128, 8: 256}
        channels = {int(scale): int(value) for scale, value in raw_channels.items()}
        if set(channels) != set(SCALES):
            raise ValueError("encoder_channels must define exactly scales 8, 16, and 32")
        if any(value <= 0 for value in channels.values()):
            raise ValueError("All multiscale encoder channel counts must be positive")
        self.channels = channels
        self.stem = nn.Conv2d(in_channels, channels[32], 3, padding=1)
        self.scale_32 = ResidualEncoderBlock(channels[32])
        self.down_16 = nn.Sequential(
            nn.GroupNorm(_group_count(channels[32]), channels[32]),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels[32], channels[16], 3, stride=2, padding=1),
        )
        self.scale_16 = ResidualEncoderBlock(channels[16])
        self.down_8 = nn.Sequential(
            nn.GroupNorm(_group_count(channels[16]), channels[16]),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels[16], channels[8], 3, stride=2, padding=1),
        )
        self.scale_8 = ResidualEncoderBlock(channels[8])

    @staticmethod
    def _mask_padding(images: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("reference_images must have shape [B,C,H,W]")
        if lengths.ndim != 1 or len(lengths) != len(images):
            raise ValueError("reference_lengths must have shape [B]")
        if images.shape[-2] != 32:
            raise ValueError(f"Reference image height must be 32, got {images.shape[-2]}")
        if bool((lengths <= 0).any()) or bool((lengths > images.shape[-1]).any()):
            raise ValueError("reference_lengths must be within the padded reference width")
        mask = _len2mask(lengths.to(torch.long), images.shape[-1], images.dtype)
        mask = mask[:, None, None, :]
        return images * mask + (mask - 1.0)

    def forward(self, images: torch.Tensor,
                lengths: torch.Tensor) -> dict[int, torch.Tensor]:
        images = self._mask_padding(images, lengths)
        feature_32 = self.scale_32(self.stem(images))
        feature_16 = self.scale_16(self.down_16(feature_32))
        feature_8 = self.scale_8(self.down_8(feature_16))
        return {32: feature_32, 16: feature_16, 8: feature_8}


class ScaleStyleHead(nn.Module):
    """Convert one scale into a fixed style condition using masked mean and std."""

    def __init__(self, channels: int, condition_dim: int, width_downsample: int,
                 eps: float = 1.0e-6):
        super().__init__()
        if width_downsample <= 0:
            raise ValueError("width_downsample must be positive")
        self.width_downsample = int(width_downsample)
        self.eps = float(eps)
        hidden_dim = max(int(channels), int(condition_dim))
        self.projection = nn.Sequential(
            nn.Linear(2 * int(channels), hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, int(condition_dim)),
        )

    def descriptor(self, features: torch.Tensor,
                   reference_lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(
            reference_lengths.to(torch.long) + self.width_downsample - 1,
            self.width_downsample,
            rounding_mode="floor",
        ).clamp_(min=1, max=features.shape[-1])
        width_mask = _len2mask(lengths, features.shape[-1], features.dtype)
        mask = width_mask[:, None, None, :]
        count = (lengths * features.shape[-2]).to(features.dtype).unsqueeze(1)
        mean = (features * mask).sum(dim=(2, 3)) / count
        variance = ((features - mean[:, :, None, None]).square() * mask).sum(dim=(2, 3)) / count
        std = torch.sqrt(variance.clamp_min(0.0) + self.eps)
        return torch.cat((mean, std), dim=1)

    def forward(self, features: torch.Tensor,
                reference_lengths: torch.Tensor) -> torch.Tensor:
        return self.projection(self.descriptor(features, reference_lengths))


class MultiScaleConditionGenerator(Generator):
    """HiGAN Generator with optional scale-specific reference conditioning."""

    def __init__(self, gen_model: Mapping[str, Any],
                 encoder_channels: Mapping[int | str, int] | None = None,
                 condition_dim: int = 32, gate_init: float = 0.1,
                 descriptor: str = "mean_std", norm: str = "group",
                 activation: str = "silu"):
        super().__init__(**_config_dict(gen_model))
        if descriptor != "mean_std":
            raise ValueError("Only descriptor='mean_std' is supported")
        if norm != "group" or activation != "silu":
            raise ValueError("Multiscale encoder currently requires GroupNorm and SiLU")
        if not self.hier or len(self.blocks) != 3:
            raise ValueError("multiscale_condition requires the hierarchical three-block HiGAN Generator")
        if int(condition_dim) != self.z_chunk_size:
            raise ValueError(
                f"condition_dim ({condition_dim}) must match Generator z chunk ({self.z_chunk_size})")
        self.condition_dim = int(condition_dim)
        self.reference_encoder = HierarchicalMultiScaleEncoder(
            in_channels=int(_config_dict(gen_model).get("input_nc", 1)),
            encoder_channels=encoder_channels,
        )
        self.style_heads = nn.ModuleDict({
            "32": ScaleStyleHead(self.reference_encoder.channels[32], self.condition_dim, 1),
            "16": ScaleStyleHead(self.reference_encoder.channels[16], self.condition_dim, 2),
            "8": ScaleStyleHead(self.reference_encoder.channels[8], self.condition_dim, 4),
        })
        self.scale_gates = nn.ParameterDict({
            str(scale): nn.Parameter(torch.tensor(float(gate_init))) for scale in SCALES
        })

    def encode_reference(self, reference_images: torch.Tensor,
                         reference_lengths: torch.Tensor) -> dict[int, torch.Tensor]:
        features = self.reference_encoder(reference_images, reference_lengths)
        return {
            scale: self.style_heads[str(scale)](features[scale], reference_lengths)
            for scale in SCALES
        }

    def gate_values(self) -> dict[str, float]:
        return {scale: float(gate.detach().cpu()) for scale, gate in self.scale_gates.items()}

    def forward(self, z: torch.Tensor, y: torch.Tensor, y_lens: torch.Tensor,
                reference_images: torch.Tensor | None = None,
                reference_lengths: torch.Tensor | None = None,
                reference_context: Mapping[int, torch.Tensor] | None = None):
        if reference_images is not None and reference_context is not None:
            raise ValueError("Pass reference_images or reference_context, not both")
        if reference_images is None and reference_context is None:
            return super().forward(z, y, y_lens)
        if reference_context is None:
            if reference_lengths is None:
                raise ValueError("reference_lengths is required with reference_images")
            reference_context = self.encode_reference(reference_images, reference_lengths)
        missing = set(SCALES) - set(reference_context)
        if missing:
            raise ValueError(f"reference_context is missing scales: {sorted(missing)}")
        chunks = list(torch.split(z, self.z_chunk_size, dim=1))
        if len(chunks) != len(self.blocks) + 1:
            raise ValueError("Latent tensor does not match the Generator hierarchical layout")
        for chunk_index, scale in enumerate(SCALES, start=1):
            condition = reference_context[scale]
            if condition.shape != chunks[chunk_index].shape:
                raise ValueError(
                    f"Scale {scale} condition shape {tuple(condition.shape)} does not match "
                    f"latent chunk {tuple(chunks[chunk_index].shape)}")
            chunks[chunk_index] = (
                chunks[chunk_index] + self.scale_gates[str(scale)] * condition)
        return super().forward(torch.cat(chunks, dim=1), y, y_lens)


def _multiscale_config(cfg) -> dict:
    architecture = getattr(cfg, "architecture", None)
    if architecture is None or not hasattr(architecture, "multiscale_condition"):
        raise ValueError("architecture.multiscale_condition is required for multiscale_condition")
    config = _config_dict(architecture.multiscale_condition)
    channels = _config_dict(config.get("encoder_channels", {32: 64, 16: 128, 8: 256}))
    config["encoder_channels"] = {int(scale): int(value) for scale, value in channels.items()}
    return config


def build_generator(cfg) -> Generator:
    variant = generator_variant(cfg)
    if variant == BASELINE_VARIANT:
        return Generator(**cfg.GenModel)
    return MultiScaleConditionGenerator(cfg.GenModel, **_multiscale_config(cfg))


def encode_reference_context(generator: Generator, reference_images: torch.Tensor,
                             reference_lengths: torch.Tensor) -> dict[int, torch.Tensor] | None:
    if isinstance(generator, MultiScaleConditionGenerator):
        return generator.encode_reference(reference_images, reference_lengths)
    return None


def slice_reference_context(reference_context: Mapping[int, torch.Tensor] | None,
                            count: int) -> dict[int, torch.Tensor] | None:
    if reference_context is None:
        return None
    return {scale: tensor[:count] for scale, tensor in reference_context.items()}


def generate_with_reference(generator: Generator, z: torch.Tensor, y: torch.Tensor,
                            y_lens: torch.Tensor, reference_images: torch.Tensor | None = None,
                            reference_lengths: torch.Tensor | None = None,
                            reference_context: Mapping[int, torch.Tensor] | None = None):
    if isinstance(generator, MultiScaleConditionGenerator):
        return generator(
            z, y, y_lens,
            reference_images=reference_images,
            reference_lengths=reference_lengths,
            reference_context=reference_context,
        )
    return generator(z, y, y_lens)


def generator_architecture_metadata(cfg) -> dict:
    variant = generator_variant(cfg)
    metadata = {"variant": variant}
    if variant == MULTISCALE_VARIANT:
        config = _multiscale_config(cfg)
        metadata["multiscale_condition"] = {
            "encoder_channels": {str(scale): value
                                 for scale, value in sorted(config["encoder_channels"].items())},
            "descriptor": str(config.get("descriptor", "mean_std")),
            "condition_dim": int(config.get("condition_dim", 32)),
            "gate_init": float(config.get("gate_init", 0.1)),
            "norm": str(config.get("norm", "group")),
            "activation": str(config.get("activation", "silu")),
        }
    return metadata


def validate_generator_architecture(saved: dict | None, expected: dict) -> None:
    if saved is None:
        if expected.get("variant") != BASELINE_VARIANT:
            raise ValueError("Legacy checkpoint has no multiscale generator architecture metadata")
        return
    if saved != expected:
        raise ValueError("Checkpoint generator architecture does not match this run")
