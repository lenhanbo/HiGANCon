from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from higan_contrastive.hdf5_data import HIGAN_HDF5_ALIASES, HIGAN_HDF5_FILES
from scripts.download_hf_checkpoint import parse_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the four original HiGAN HDF5 files")
    parser.add_argument("source", help="Checkpoint HF URL, hf:// URI, or owner/repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prefix", default="",
                        help="Optional folder containing the HDF5 files in the Hub repo")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_id, _, source_revision = parse_source(args.source, "latest.pt", args.revision)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key, basename in HIGAN_HDF5_FILES.items():
        candidates = HIGAN_HDF5_ALIASES.get(key, (basename,))
        last_error = None
        for candidate in candidates:
            filename = (f"{args.prefix.strip('/')}/{candidate}"
                        if args.prefix.strip("/") else candidate)
            try:
                cached = Path(hf_hub_download(
                    repo_id=repo_id, filename=filename, revision=source_revision))
                shutil.copy2(cached, output / basename)
                break
            except EntryNotFoundError as error:
                last_error = error
        else:
            raise last_error or FileNotFoundError(basename)
    print(output.resolve())


if __name__ == "__main__":
    main()
