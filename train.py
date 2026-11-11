import argparse

from higan_contrastive.trainer import HiGANTrainer
from lib.utils import yaml2config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline or generated-style SupCon HiGAN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    trainer = HiGANTrainer(yaml2config(args.config), args.config, args.dataset_root,
                           args.manifest_root, args.output, args.resume, args.max_hours)
    trainer.train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()

