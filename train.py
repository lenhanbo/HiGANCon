import argparse

from higan_contrastive.ctc_trainer import CTCSoftTrainer
from higan_contrastive.trainer import HiGANTrainer
from lib.utils import yaml2config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline or generated-style SupCon HiGAN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-format", choices=("iam_raw", "higan_hdf5"), default="iam_raw")
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop at this global step and save last.pt; 0 disables the limit")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--contrastive-start-step", type=int, default=None)
    parser.add_argument("--contrastive-ramp-steps", type=int, default=None)
    parser.add_argument("--preview-every-steps", type=int, default=None,
                        help="Runtime preview interval; 0 disables previews without changing config hash")
    parser.add_argument("--no", action="append", choices=("supcon", "ctc-guided"), default=[],
                        metavar="LOSS", help="Disable supcon or ctc-guided (repeatable)")
    parser.add_argument("--no-supcon", action="store_true", help="Alias for --no supcon")
    parser.add_argument("--no-ctc-guided", action="store_true",
                        help="Alias for --no ctc-guided")
    return parser


def reject_ctc_schedule_overrides(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    forbidden = {
        "--contrastive-start-step": args.contrastive_start_step,
        "--contrastive-ramp-steps": args.contrastive_ramp_steps,
        "--preview-every-steps": args.preview_every_steps,
    }
    used = [name for name, value in forbidden.items() if value is not None]
    if used:
        parser.error(f"{', '.join(used)} cannot be used with a ctc_match config; edit YAML instead")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = yaml2config(args.config)
    disabled = set(args.no)
    if args.no_supcon:
        disabled.add("supcon")
    if args.no_ctc_guided:
        disabled.add("ctc-guided")
    if hasattr(cfg, "ctc_match"):
        reject_ctc_schedule_overrides(parser, args)
        trainer = CTCSoftTrainer(
            cfg, args.config, args.dataset_root, args.manifest_root, args.output,
            args.resume, args.max_hours, args.max_steps, dataset_format=args.dataset_format,
            enable_supcon="supcon" not in disabled,
            enable_ctc_guided="ctc-guided" not in disabled)
    else:
        if "ctc-guided" in disabled:
            parser.error("--no ctc-guided requires a config containing ctc_match")
        trainer = HiGANTrainer(
            cfg, args.config, args.dataset_root, args.manifest_root, args.output,
            args.resume, args.max_hours, args.max_steps, args.contrastive_start_step,
            args.contrastive_ramp_steps, args.dataset_format, args.preview_every_steps,
            enable_supcon="supcon" not in disabled)
    trainer.train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()

