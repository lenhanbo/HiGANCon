import argparse

from higan_contrastive.trainer import HiGANTrainer
from lib.utils import yaml2config


def main() -> None:
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
    args = parser.parse_args()
    trainer = HiGANTrainer(yaml2config(args.config), args.config, args.dataset_root,
                           args.manifest_root, args.output, args.resume, args.max_hours, args.max_steps,
                           args.contrastive_start_step, args.contrastive_ramp_steps,
                           args.dataset_format, args.preview_every_steps)
    trainer.train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()

