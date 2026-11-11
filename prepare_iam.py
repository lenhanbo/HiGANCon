import argparse
import json

from higan_contrastive.data import write_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic writer-disjoint IAM manifests")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(write_manifests(args.dataset_root, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
