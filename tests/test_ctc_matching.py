import itertools

import pytest
import torch

from higan_contrastive.ctc_matching import (ctc_forward_backward, horizontal_patches,
                                            interpolate_alignment, soft_patch_cosine)
from higan_contrastive.ctc_preview import save_alignment_heatmap, save_ctc_preview


def _collapse(path):
    result = []
    previous = None
    for symbol in path:
        if symbol != previous and symbol != 0:
            result.append(symbol)
        previous = symbol
    return result


@pytest.mark.parametrize("target", ([1, 2], [1, 1]))
def test_ctc_posterior_matches_brute_force(target):
    torch.manual_seed(3)
    logits = torch.randn(1, 4, 3)
    alignment = ctc_forward_backward(
        logits, torch.tensor([target]), torch.tensor([4]), torch.tensor([len(target)]))
    probabilities = logits.log_softmax(-1).exp()[0]
    occurrence = torch.zeros(len(target), 4)
    total = 0.0
    for path in itertools.product(range(3), repeat=4):
        if _collapse(path) != list(target):
            continue
        probability = float(torch.prod(probabilities[range(4), torch.tensor(path)]))
        total += probability
        occurrence_index, previous = -1, None
        for time_index, symbol in enumerate(path):
            if symbol != previous and symbol != 0:
                occurrence_index += 1
            if symbol != 0:
                occurrence[occurrence_index, time_index] += probability
            previous = symbol
    occurrence /= total
    occurrence /= occurrence.sum(dim=-1, keepdim=True)
    assert alignment.valid.tolist() == [True]
    assert alignment.log_probability.item() == pytest.approx(torch.tensor(total).log().item(), abs=1e-5)
    assert torch.allclose(alignment.weights[0, :len(target)], occurrence, atol=1e-5)


def test_ctc_invalid_path_is_masked_without_nan():
    alignment = ctc_forward_backward(
        torch.randn(1, 2, 3), torch.tensor([[1, 1]]), torch.tensor([2]), torch.tensor([2]))
    assert not alignment.valid.item()
    assert alignment.weights.sum().item() == 0
    assert not torch.isnan(alignment.weights).any()


def test_interpolation_normalization_and_padding_mask():
    weights = torch.tensor([[[0.2, 0.3, 0.5, 0.0], [0.5, 0.5, 0.0, 0.0]]])
    resized, mask = interpolate_alignment(weights, torch.tensor([3]), 8, source_width=4)
    assert mask.tolist() == [[True] * 6 + [False] * 2]
    assert torch.allclose(resized.sum(-1), torch.ones(1, 2))
    assert resized[..., 6:].sum() == 0


def test_soft_patch_gradient_only_reaches_generated_branch():
    real = torch.randn(1, 2, 2, 4, requires_grad=True)
    fake = torch.randn(1, 2, 2, 4, requires_grad=True)
    weights = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    loss, _, _ = soft_patch_cosine(
        real, fake, weights, weights, torch.tensor([[True]]), torch.tensor([True]))
    loss.backward()
    assert real.grad is None
    assert fake.grad is not None and fake.grad.abs().sum() > 0
    assert horizontal_patches(fake.detach(), 3).shape[:2] == (1, 4)


def test_complete_preview_grid_heatmap_and_json(tmp_path):
    images = torch.full((2, 1, 32, 48), -1.0)
    images[:, :, 8:24, 10:30] = 1.0
    output = tmp_path / "step_00000200.png"
    save_ctc_preview(
        images, torch.tensor([32, 48]), images, torch.tensor([32, 48]),
        images, torch.tensor([32, 48]), images, torch.tensor([32, 48]),
        ["ref-a", "ref-b"], ["query-a", "query-b"], output,
        writer_ids=["w0", "w1"], supcon_images=images,
        supcon_lengths=torch.tensor([32, 48]), supcon_text="shared",
        metadata={"active_losses": ["higan", "supcon", "ctc-guided"]})
    heatmap = tmp_path / "alignments" / "step_00000200_item_00.png"
    save_alignment_heatmap(torch.tensor([[0.2, 0.8], [0.7, 0.3]]), "ab", heatmap,
                           torch.tensor([0.5, 0.6]), torch.tensor([1, 0]))
    assert output.exists() and output.with_suffix(".json").exists() and heatmap.exists()
