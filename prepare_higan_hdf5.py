import argparse
import json

from higan_contrastive.hdf5_data import (prepare_higan_hdf5_metadata,
                                         prepare_spisgan_hdf5_metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate handwriting HDF5 files and create metadata")
    parser.add_argument("--hdf5-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-format", choices=("higan_hdf5", "spisgan_hdf5"),
                        default="higan_hdf5")
    args = parser.parse_args()
    prepare = (prepare_spisgan_hdf5_metadata if args.dataset_format == "spisgan_hdf5"
               else prepare_higan_hdf5_metadata)
    print(json.dumps(prepare(args.hdf5_root, args.output), indent=2))


if __name__ == "__main__":
    main()
