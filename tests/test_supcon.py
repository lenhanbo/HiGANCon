import torch
import pytest
from torch import nn

from higan_contrastive.losses import (contrastive_weight_at_step, optional_contrastive_loss,
                                      supervised_contrastive_loss)


def test_supcon_finite_and_separates_pairs():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]],
                            requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(features, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_generated_supcon_reaches_generator_and_encoder():
    encoder = nn.Linear(5, 3)
    generator = nn.Linear(5, 5)
    references = torch.randn(4, 5)
    labels = torch.tensor([0, 0, 1, 1])
    style = encoder(references)
    generated = generator(torch.cat((torch.zeros(4, 2), style), dim=1))
    generated_style = encoder(generated)
    supervised_contrastive_loss(generated_style, labels).backward()
    assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0
    assert generator.weight.grad is not None and generator.weight.grad.abs().sum() > 0


def test_zero_weight_does_not_build_contrastive_path():
    called = False

    def forbidden():
        nonlocal called
        called = True
        raise AssertionError("baseline must not call the generated-style branch")

    result = optional_contrastive_loss(0.0, torch.tensor(1.0), forbidden)
    assert result.item() == 0.0
    assert not called


def test_step_based_contrastive_schedule():
    assert contrastive_weight_at_step(99, 0.1, 100, 10) == 0.0
    assert contrastive_weight_at_step(100, 0.1, 100, 10) == pytest.approx(0.01)
    assert contrastive_weight_at_step(109, 0.1, 100, 10) == pytest.approx(0.1)
    assert contrastive_weight_at_step(500, 0.1, 100, 0) == pytest.approx(0.1)
    assert contrastive_weight_at_step(500, 0.0, 100, 10) == 0.0
