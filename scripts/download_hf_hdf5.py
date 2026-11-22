from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Support the documented direct invocation: python scripts/download_hf_hdf5.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from higan_contrastive.hdf5_data import (HIGAN_HDF5_ALIASES, HIGAN_HDF5_FILES,
                                         SPISGAN_HDF5_FILES)
from scripts.download_hf_checkpoint import infer_repo_type, parse_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HiGAN or SpiS-GAN IAM HDF5 files")
    parser.add_argument("source", help="Checkpoint HF URL, hf:// URI, or owner/repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--repo-type", choices=("auto", "model", "dataset", "space"), default="auto",
        help="Hub repository type; auto detects full /datasets/ and /spaces/ URLs",
    )
    parser.add_argument("--prefix", default="",
                        help="Optional folder containing the HDF5 files in the Hub repo")
    parser.add_argument("--dataset-format", choices=("higan_hdf5", "spisgan_hdf5"),
                        default="higan_hdf5")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_id, _, source_revision = parse_source(args.source, "latest.pt", args.revision)
    repo_type = infer_repo_type(args.source) if args.repo_type == "auto" else args.repo_type
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = SPISGAN_HDF5_FILES if args.dataset_format == "spisgan_hdf5" else HIGAN_HDF5_FILES
    for key, basename in files.items():
        candidates = (basename,) if args.dataset_format == "spisgan_hdf5" else HIGAN_HDF5_ALIASES.get(key, (basename,))
        last_error = None
        for candidate in candidates:
            filename = (f"{args.prefix.strip('/')}/{candidate}"
                        if args.prefix.strip("/") else candidate)
            try:
                cached = Path(hf_hub_download(
                    repo_id=repo_id, filename=filename, revision=source_revision,
                    repo_type=repo_type))
                shutil.copy2(cached, output / basename)
                break
            except EntryNotFoundError as error:
                last_error = error
        else:
            raise last_error or FileNotFoundError(basename)
    print(output.resolve())


if __name__ == "__main__":
    main()
