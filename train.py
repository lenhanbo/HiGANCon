import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import torch

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
    parser.add_argument("--no-eval", action="store_true",
                        help="Do not automatically evaluate after training finishes")
    parser.add_argument("--eval-max-samples", type=int, default=None,
                        help="Automatic eval sample limit; 0 evaluates the full test split")
    parser.add_argument("--eval-fid-kid-protocol", choices=("higan", "torchvision"),
                        default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-skip-hwd", action="store_true", default=None,
                        help="Skip HWD during automatic evaluation")
    parser.add_argument("--eval-hwd-checkpoint", default=None)
    parser.add_argument("--eval-recognizer-checkpoint", default=None)
    parser.add_argument("--eval-preview-samples", type=int, default=None)
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


def automatic_evaluation_args(args: argparse.Namespace, checkpoint: Path) -> SimpleNamespace:
    """Translate training CLI settings to the existing evaluator interface."""
    return SimpleNamespace(
        config=args.config,
        dataset_root=args.dataset_root,
        dataset_format=args.dataset_format,
        manifest_root=args.manifest_root,
        checkpoint=str(checkpoint),
        recognizer_checkpoint=args.eval_recognizer_checkpoint,
        output=str(Path(args.output).resolve() / "evaluation.json"),
        max_samples=args.eval_max_samples,
        fid_kid_protocol=args.eval_fid_kid_protocol,
        eval_seed=args.eval_seed,
        log_every=10,
        preview_samples=args.eval_preview_samples,
        preview_page_size=16,
        preview_dir=None,
        preview_query_mode="dataset",
        preview_query_text=None,
        hwd_checkpoint=args.eval_hwd_checkpoint,
        skip_hwd=args.eval_skip_hwd,
    )


def run_automatic_evaluation(args: argparse.Namespace) -> None:
    checkpoint = Path(args.output).resolve() / "checkpoints" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Training finished but automatic evaluation checkpoint is missing: {checkpoint}")
    from evaluate import evaluate
    print(f"[train] finished; starting automatic evaluation from {checkpoint}", flush=True)
    evaluate(automatic_evaluation_args(args, checkpoint))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = yaml2config(args.config)
    evaluation_cfg = cfg.evaluation
    if args.eval_max_samples is None:
        args.eval_max_samples = int(evaluation_cfg.max_samples)
    if args.eval_fid_kid_protocol is None:
        args.eval_fid_kid_protocol = str(evaluation_cfg.fid_kid_protocol)
    if args.eval_seed is None:
        args.eval_seed = int(evaluation_cfg.seed)
    if args.eval_skip_hwd is None:
        args.eval_skip_hwd = bool(evaluation_cfg.skip_hwd)
    if args.eval_preview_samples is None:
        args.eval_preview_samples = int(evaluation_cfg.preview_samples)
    if args.eval_max_samples < 0:
        parser.error("--eval-max-samples must be >= 0")
    if args.eval_preview_samples < -1:
        parser.error("--eval-preview-samples must be -1, 0, or a positive integer")
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
    if args.smoke_test:
        print("[train] smoke test complete; automatic evaluation skipped", flush=True)
    elif args.no_eval:
        print("[train] automatic evaluation disabled by --no-eval", flush=True)
    elif not bool(evaluation_cfg.enabled):
        print("[train] automatic evaluation disabled by config", flush=True)
    else:
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_automatic_evaluation(args)


if __name__ == "__main__":
    main()

