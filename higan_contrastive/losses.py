from __future__ import annotations

import torch
import torch.nn.functional as F
from collections.abc import Callable


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean()


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss over one normalized feature vector per view."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("Expected features [N,D] and labels [N]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    if not positive.any(dim=1).all():
        raise ValueError("Every anchor must have at least one positive")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = logits.exp().masked_fill(eye, 0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return -(log_prob * positive).sum(dim=1).div(positive.sum(dim=1)).mean()


def contrastive_weight_at_step(step_one_based: int, target: float, start_step: int,
                               ramp_steps: int) -> float:
    """Step schedule: zero before start_step, then linear ramp to target."""
    if step_one_based < 1 or start_step < 1:
        raise ValueError("step_one_based and start_step must be >= 1")
    if target <= 0 or step_one_based < start_step:
        return 0.0
    if ramp_steps <= 1:
        return float(target)
    progress = min(1.0, (step_one_based - start_step + 1) / ramp_steps)
    return float(target) * progress


def optional_contrastive_loss(weight: float, reference: torch.Tensor,
                              compute: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Avoid constructing any contrastive graph for the exact baseline path."""
    return reference.new_zeros(()) if weight <= 0 else compute()
