from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from higan_contrastive.checkpoint import (file_hash,
                                          legacy_config_hash_without_architecture,
                                          legacy_config_hash_without_evaluation_setup)
from networks.BigGAN_networks import Generator
from networks.multiscale_generator import (
    MultiScaleConditionGenerator, ScaleStyleHead, build_generator,
    encode_reference_context, generate_with_reference,
    generator_architecture_metadata, validate_generator_architecture,
)


def generator_config():
    return {
        "G_ch": 2,
        "style_dim": 128,
        "bottom_width": 4,
        "bottom_height": 4,
        "resolution": 32,
        "G_kernel_size": 3,
        "G_attn": "64",
        "n_class": 72,
        "G_shared": False,
        "no_hier": False,
        "G_activation": torch.nn.ReLU(inplace=False),
        "init": "none",
        "G_param": "none",
        "norm_style": "bn",
        "bn_linear": "SN",
        "input_nc": 1,
        "one_hot": True,
        "first_layer": True,
        "one_hot_k": 1,
    }


def config(variant="baseline"):
    return SimpleNamespace(
        GenModel=generator_config(),
        architecture=SimpleNamespace(
            generator_variant=variant,
            multiscale_condition={
                "encoder_channels": {32: 4, 16: 8, 8: 16},
                "descriptor": "mean_std",
                "condition_dim": 32,
                "gate_init": 0.1,
                "norm": "group",
                "activation": "silu",
            },
        ),
    )


def generator_inputs(batch_size=2):
    z = torch.randn(batch_size, 128)
    labels = torch.tensor([[1, 2, 3], [4, 5, 0]])[:batch_size]
    label_lens = torch.tensor([3, 2], dtype=torch.int32)[:batch_size]
    references = torch.randn(batch_size, 1, 32, 32)
    reference_lens = torch.tensor([32, 20], dtype=torch.int32)[:batch_size]
    return z, labels, label_lens, references, reference_lens


def test_baseline_factory_returns_exact_original_generator():
    torch.manual_seed(123)
    direct = Generator(**generator_config())
    torch.manual_seed(123)
    model = build_generator(config("baseline"))
    assert type(model) is Generator
    assert not any(key.startswith("reference_encoder.") for key in model.state_dict())
    for key, value in direct.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_random_forward_skips_multiscale_encoder():
    model = build_generator(config("multiscale_condition")).eval()
    z, labels, label_lens, _, _ = generator_inputs()
    with patch.object(model.reference_encoder, "forward",
                      side_effect=AssertionError("reference encoder was called")):
        output = model(z, labels, label_lens)
    assert output.shape[0] == len(z)
    assert output.shape[-2] == 32


def test_masked_mean_std_matches_manual_calculation():
    head = ScaleStyleHead(channels=1, condition_dim=2, width_downsample=1, eps=0.0)
    features = torch.tensor([[[[1.0, 3.0, 100.0, 200.0],
                               [5.0, 7.0, 300.0, 400.0]]]])
    descriptor = head.descriptor(features, torch.tensor([2]))
    valid = torch.tensor([1.0, 3.0, 5.0, 7.0])
    expected = torch.stack((valid.mean(), valid.std(unbiased=False))).unsqueeze(0)
    torch.testing.assert_close(descriptor, expected)


def test_reference_padding_does_not_change_context():
    model = build_generator(config("multiscale_condition")).eval()
    _, _, _, references, reference_lens = generator_inputs()
    changed = references.clone()
    changed[1, :, :, 20:] = torch.randn_like(changed[1, :, :, 20:]) * 100
    first = model.encode_reference(references, reference_lens)
    second = model.encode_reference(changed, reference_lens)
    for scale in (8, 16, 32):
        assert first[scale].shape == (2, 32)
        torch.testing.assert_close(first[scale], second[scale])


def test_zero_gates_recover_original_generator_output():
    torch.manual_seed(7)
    baseline = Generator(**generator_config()).eval()
    torch.manual_seed(99)
    multiscale = build_generator(config("multiscale_condition")).eval()
    load_result = multiscale.load_state_dict(baseline.state_dict(), strict=False)
    assert not load_result.unexpected_keys
    for gate in multiscale.scale_gates.values():
        gate.data.zero_()
    z, labels, label_lens, references, reference_lens = generator_inputs()
    expected = baseline(z, labels, label_lens)
    actual = generate_with_reference(
        multiscale, z, labels, label_lens, references, reference_lens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cached_and_direct_reference_context_are_equivalent_and_trainable():
    model = build_generator(config("multiscale_condition")).eval()
    z, labels, label_lens, references, reference_lens = generator_inputs()
    context = encode_reference_context(model, references, reference_lens)
    direct = generate_with_reference(
        model, z, labels, label_lens, references, reference_lens)
    cached = generate_with_reference(
        model, z, labels, label_lens, reference_context=context)
    torch.testing.assert_close(cached, direct)
    cached.mean().backward()
    assert all(gate.grad is not None for gate in model.scale_gates.values())
    assert any(parameter.grad is not None for parameter in model.reference_encoder.parameters())


def test_checkpoint_architecture_rejects_variant_mismatch():
    baseline = generator_architecture_metadata(config("baseline"))
    multiscale = generator_architecture_metadata(config("multiscale_condition"))
    validate_generator_architecture(None, baseline)
    validate_generator_architecture(multiscale, multiscale)
    with pytest.raises(ValueError, match="architecture"):
        validate_generator_architecture(baseline, multiscale)
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        validate_generator_architecture(None, multiscale)


def test_legacy_baseline_config_hash_ignores_only_appended_architecture(tmp_path):
    old_config = tmp_path / "old.yaml"
    new_config = tmp_path / "new.yaml"
    old_config.write_text("seed: 1\ntraining:\n  epochs: 2\n", encoding="utf-8")
    new_config.write_text(
        old_config.read_text(encoding="utf-8")
        + "\narchitecture:\n  generator_variant: baseline\n",
        encoding="utf-8",
    )
    assert legacy_config_hash_without_architecture(new_config) == file_hash(old_config)


def test_legacy_config_hash_ignores_only_evaluation_setup(tmp_path):
    old_config = tmp_path / "old.yaml"
    new_config = tmp_path / "new.yaml"
    old_config.write_text(
        "evaluation:\n  enabled: true\n  max_samples: 0\n", encoding="utf-8")
    new_config.write_text(
        "evaluation:\n  enabled: true\n  setup: higan\n  max_samples: 0\n", encoding="utf-8")
    assert legacy_config_hash_without_evaluation_setup(new_config) == file_hash(old_config)
