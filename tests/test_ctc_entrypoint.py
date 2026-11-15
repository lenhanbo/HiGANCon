import pytest
from pathlib import Path

from higan_contrastive.ctc_trainer import paired_indices, validate_bidirectional_pairs
from train import automatic_evaluation_args, build_parser, reject_ctc_schedule_overrides


def _required_args():
    return ["--config", "x.yml", "--dataset-root", "data", "--manifest-root", "manifest",
            "--output", "out"]


def test_default_and_all_loss_opt_out_spellings():
    parser = build_parser()
    default = parser.parse_args(_required_args())
    assert default.no == [] and not default.no_supcon and not default.no_ctc_guided
    spaced = parser.parse_args(_required_args() + ["--no", "supcon", "--no", "ctc-guided"])
    assert spaced.no == ["supcon", "ctc-guided"]
    aliases = parser.parse_args(_required_args() + ["--no-supcon", "--no-ctc-guided"])
    assert aliases.no_supcon and aliases.no_ctc_guided


def test_two_supcon_text_mode_flags_are_mutually_exclusive():
    parser = build_parser()
    shared = parser.parse_args(_required_args() + ["--supcon-shared-text"])
    per_item = parser.parse_args(_required_args() + ["--supcon-per-item-text"])
    assert shared.supcon_text_mode == "shared"
    assert per_item.supcon_text_mode == "per_item"
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args()
                          + ["--supcon-shared-text", "--supcon-per-item-text"])


@pytest.mark.parametrize("option", ("--contrastive-start-step", "--contrastive-ramp-steps",
                                    "--preview-every-steps"))
def test_ctc_config_rejects_schedule_overrides(option):
    parser = build_parser()
    args = parser.parse_args(_required_args() + [option, "1"])
    with pytest.raises(SystemExit):
        reject_ctc_schedule_overrides(parser, args)


def test_bidirectional_pairing_and_validation():
    assert paired_indices(4).tolist() == [1, 0, 3, 2]
    batch = {"writer_ids": ["a", "a", "b", "b"],
             "word_ids": ["a1", "a2", "b1", "b2"]}
    validate_bidirectional_pairs(batch)
    batch["writer_ids"][1] = "x"
    with pytest.raises(ValueError, match="same writer"):
        validate_bidirectional_pairs(batch)


def test_automatic_evaluation_reuses_training_inputs(tmp_path):
    args = build_parser().parse_args(_required_args() + [
        "--eval-max-samples", "12", "--eval-skip-hwd"])
    args.output = str(tmp_path / "run")
    checkpoint = Path(args.output) / "checkpoints" / "last.pt"
    evaluation = automatic_evaluation_args(args, checkpoint)
    assert evaluation.checkpoint == str(checkpoint)
    assert evaluation.output == str((tmp_path / "run" / "evaluation.json").resolve())
    assert evaluation.max_samples == 12
    assert evaluation.skip_hwd
