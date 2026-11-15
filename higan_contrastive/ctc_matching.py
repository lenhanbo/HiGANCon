from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CTCAlignment:
    """Conditional alignment of each target occurrence to CTC timesteps."""

    weights: torch.Tensor       # [B, U, T], normalized over T
    occurrence_mask: torch.Tensor  # [B, U]
    valid: torch.Tensor         # [B]
    log_probability: torch.Tensor  # [B]
    entropy: torch.Tensor       # [B, U]
    peaks: torch.Tensor         # [B, U]


def _as_batch_log_probs(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("CTC logits must have shape [T,B,C] or [B,T,C]")
    if logits.shape[0] == batch_size:
        batch_logits = logits
    elif logits.shape[1] == batch_size:
        batch_logits = logits.transpose(0, 1)
    else:
        raise ValueError("Cannot infer the batch dimension of CTC logits")
    return batch_logits.float().log_softmax(dim=-1)


def ctc_forward_backward(logits: torch.Tensor, targets: torch.Tensor,
                         input_lengths: torch.Tensor, target_lengths: torch.Tensor,
                         blank: int = 0, eps: float = 1e-8) -> CTCAlignment:
    """Compute occurrence-level CTC posteriors in log-space FP32.

    Repeated labels are handled by prohibiting the usual two-state skip. Invalid
    paths remain zero and are explicitly marked instead of producing NaNs.
    """
    if targets.ndim != 2:
        raise ValueError("targets must have shape [B,U]")
    batch_size, max_targets = targets.shape
    log_probs = _as_batch_log_probs(logits, batch_size)
    max_time = log_probs.shape[1]
    device = log_probs.device
    weights = torch.zeros(batch_size, max_targets, max_time, device=device)
    occurrence_mask = (torch.arange(max_targets, device=device)[None, :]
                       < target_lengths.to(device)[:, None])
    valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    log_probability = torch.full((batch_size,), -torch.inf, device=device)

    neg_inf = torch.tensor(-torch.inf, device=device)
    for batch_index in range(batch_size):
        time_count = min(int(input_lengths[batch_index]), max_time)
        target_count = min(int(target_lengths[batch_index]), max_targets)
        if time_count <= 0 or target_count <= 0:
            continue
        target = targets[batch_index, :target_count].to(device=device, dtype=torch.long)
        minimum_time = target_count + int((target[1:] == target[:-1]).sum().item())
        if time_count < minimum_time:
            continue
        states = torch.full((2 * target_count + 1,), blank, dtype=torch.long, device=device)
        states[1::2] = target
        state_count = len(states)
        emissions = log_probs[batch_index, :time_count, states]
        alpha = torch.full((time_count, state_count), -torch.inf, device=device)
        alpha[0, 0] = emissions[0, 0]
        if state_count > 1:
            alpha[0, 1] = emissions[0, 1]
        for time_index in range(1, time_count):
            for state_index in range(state_count):
                previous = [alpha[time_index - 1, state_index]]
                if state_index > 0:
                    previous.append(alpha[time_index - 1, state_index - 1])
                if (state_index > 1 and states[state_index] != blank
                        and states[state_index] != states[state_index - 2]):
                    previous.append(alpha[time_index - 1, state_index - 2])
                alpha[time_index, state_index] = (
                    torch.logsumexp(torch.stack(previous), dim=0) + emissions[time_index, state_index])
        endings = alpha[-1, -2:] if state_count > 1 else alpha[-1, -1:]
        log_z = torch.logsumexp(endings, dim=0)
        if not torch.isfinite(log_z):
            continue
        beta = torch.full_like(alpha, -torch.inf)
        beta[-1, -1] = 0.0
        if state_count > 1:
            beta[-1, -2] = 0.0
        for time_index in range(time_count - 2, -1, -1):
            for state_index in range(state_count):
                following = [beta[time_index + 1, state_index]
                             + emissions[time_index + 1, state_index]]
                if state_index + 1 < state_count:
                    following.append(beta[time_index + 1, state_index + 1]
                                     + emissions[time_index + 1, state_index + 1])
                if (state_index + 2 < state_count and states[state_index] != blank
                        and states[state_index] != states[state_index + 2]):
                    following.append(beta[time_index + 1, state_index + 2]
                                     + emissions[time_index + 1, state_index + 2])
                beta[time_index, state_index] = torch.logsumexp(torch.stack(following), dim=0)
        occurrence = (alpha[:, 1::2] + beta[:, 1::2] - log_z).exp().transpose(0, 1)
        occurrence = occurrence / occurrence.sum(dim=1, keepdim=True).clamp_min(eps)
        weights[batch_index, :target_count, :time_count] = occurrence
        valid[batch_index] = True
        log_probability[batch_index] = log_z

    safe_weights = weights.clamp_min(eps)
    entropy = -(weights * safe_weights.log()).sum(dim=-1)
    entropy = entropy.masked_fill(~occurrence_mask, 0.0)
    peaks = weights.argmax(dim=-1).masked_fill(~occurrence_mask, 0)
    return CTCAlignment(weights, occurrence_mask, valid, log_probability, entropy, peaks)


def interpolate_alignment(weights: torch.Tensor, input_lengths: torch.Tensor,
                          output_width: int, source_width: int | None = None,
                          eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly resize occurrence weights and mask padded feature positions."""
    if weights.ndim != 3 or output_width <= 0:
        raise ValueError("weights must be [B,U,T] and output_width must be positive")
    batch_size, _, max_time = weights.shape
    source_width = max_time if source_width is None else int(source_width)
    resized = torch.zeros(batch_size, weights.shape[1], output_width,
                          device=weights.device, dtype=weights.dtype)
    output_lengths = torch.zeros(batch_size, dtype=torch.long, device=weights.device)
    for index in range(batch_size):
        valid_source = min(max_time, int(input_lengths[index]))
        if valid_source <= 0:
            continue
        valid_output = max(1, min(output_width, round(valid_source * output_width / source_width)))
        output_lengths[index] = valid_output
        sample = F.interpolate(weights[index:index + 1, :, :valid_source], size=valid_output,
                               mode="linear", align_corners=False)
        resized[index, :, :valid_output] = sample[0]
    mask = torch.arange(output_width, device=weights.device)[None, :] < output_lengths[:, None]
    resized = resized * mask[:, None, :]
    resized = resized / resized.sum(dim=-1, keepdim=True).clamp_min(eps)
    return resized, mask


def horizontal_patches(features: torch.Tensor, patch_width: int = 3) -> torch.Tensor:
    """Return full-height, centered horizontal patches as [B,W,D]."""
    if features.ndim != 4 or patch_width < 1 or patch_width % 2 == 0:
        raise ValueError("features must be [B,C,H,W] and patch_width must be positive and odd")
    padded = F.pad(features, (patch_width // 2, patch_width // 2, 0, 0))
    patches = padded.unfold(3, patch_width, 1).permute(0, 3, 1, 2, 4)
    return patches.flatten(2)


def soft_patch_cosine(real_features: torch.Tensor, generated_features: torch.Tensor,
                      real_weights: torch.Tensor, generated_weights: torch.Tensor,
                      occurrence_mask: torch.Tensor, valid: torch.Tensor,
                      patch_width: int = 3, eps: float = 1e-8
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Positive cosine loss between posterior-weighted real/fake patches."""
    real_patches = F.normalize(horizontal_patches(real_features, patch_width), dim=-1, eps=eps)
    fake_patches = F.normalize(horizontal_patches(generated_features, patch_width), dim=-1, eps=eps)
    real_soft = torch.bmm(real_weights, real_patches).detach()
    fake_soft = torch.bmm(generated_weights, fake_patches)
    cosine = F.cosine_similarity(real_soft, fake_soft, dim=-1, eps=eps)
    pair_mask = occurrence_mask & valid[:, None]
    count = pair_mask.sum()
    loss = ((1.0 - cosine) * pair_mask).sum() / count.clamp_min(1)
    return loss, cosine, pair_mask
