from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import hf_hub_download


def infer_repo_type(source: str) -> str:
    """Infer the Hub repository type from a full URL; default to a model repo."""
    source = source.strip()
    if source.startswith(("https://", "http://")):
        parts = urlparse(source).path.strip("/").split("/")
        if parts and parts[0] == "datasets":
            return "dataset"
        if parts and parts[0] == "spaces":
            return "space"
    if source.startswith("hf://datasets/"):
        return "dataset"
    if source.startswith("hf://spaces/"):
        return "space"
    return "model"


def parse_source(source: str, default_filename: str,
                 default_revision: str) -> tuple[str, str, str]:
    source = source.strip()
    if source.startswith("hf://"):
        payload = source[5:]
        if payload.startswith(("datasets/", "spaces/", "models/")):
            payload = payload.split("/", 1)[1]
        parts = payload.split("/", 2)
        if len(parts) != 3:
            raise ValueError("hf:// source must be hf://owner/repo/path/file.pt[@revision]")
        repo_id = "/".join(parts[:2])
        filename_revision = parts[2]
        if "@" in filename_revision:
            filename, revision = filename_revision.rsplit("@", 1)
        else:
            filename, revision = filename_revision, default_revision
        return repo_id, filename, revision

    if source.startswith(("https://", "http://")):
        parsed = urlparse(source)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("Checkpoint URL must point to huggingface.co")
        parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
        if parts and parts[0] in {"datasets", "spaces", "models"}:
            parts = parts[1:]
        if len(parts) == 2 and all(parts):
            return "/".join(parts), default_filename, default_revision
        if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
            raise ValueError(
                "Expected a Hugging Face repository URL or a /resolve|blob/ file URL")
        return "/".join(parts[:2]), "/".join(parts[4:]), parts[3]

    parts = source.split("/")
    if len(parts) == 3 and parts[0] in {"datasets", "spaces", "models"}:
        parts = parts[1:]
        source = "/".join(parts)
    if len(parts) == 2 and all(parts):
        return source, default_filename, default_revision
    raise ValueError("Use a Hugging Face URL, hf:// URI, or owner/repo repository ID")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one checkpoint from Hugging Face Hub")
    parser.add_argument("source", help="HF URL, hf:// URI, or owner/repo")
    parser.add_argument("--filename", default="latest.pt",
                        help="Filename when source is only owner/repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--repo-type", choices=("auto", "model", "dataset", "space"), default="auto",
        help="Hub repository type; auto detects full /datasets/ and /spaces/ URLs",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_id, filename, revision = parse_source(args.source, args.filename, args.revision)
    repo_type = infer_repo_type(args.source) if args.repo_type == "auto" else args.repo_type
    cached = Path(hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, repo_type=repo_type))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / cached.name
    shutil.copy2(cached, destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
