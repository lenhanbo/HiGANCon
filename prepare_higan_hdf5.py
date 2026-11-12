import argparse
import json

from higan_contrastive.hdf5_data import prepare_higan_hdf5_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HiGAN HDF5 files and create metadata")
    parser.add_argument("--hdf5-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_higan_hdf5_metadata(args.hdf5_root, args.output), indent=2))


if __name__ == "__main__":
    main()
