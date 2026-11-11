from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from lib.alphabet import Alphabets, strLabelConverter


REQUIRED_IAM_FILES = ("forms.txt", "linux_gt.txt", "train_gt.txt", "val_gt.txt", "words")


def validate_dataset_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    missing = [name for name in REQUIRED_IAM_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Invalid DATASET_ROOT {root}: missing {', '.join(missing)}")
    if not (root / "words").is_dir():
        raise NotADirectoryError(root / "words")
    return root


def parse_forms(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed forms.txt line {line_no}: {raw.rstrip()}")
            form_id, writer_id = fields[0], fields[1]
            previous = mapping.setdefault(form_id, writer_id)
            if previous != writer_id:
                raise ValueError(f"Conflicting writer for form {form_id}")
    return mapping


def form_id_from_word_id(word_id: str) -> str:
    parts = word_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid IAM word id: {word_id}")
    return "-".join(parts[:-2])


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_linux_gt(root: str | Path) -> list[dict]:
    root = validate_dataset_root(root)
    form_writers = parse_forms(root / "forms.txt")
    alphabet = set(Alphabets["all"])
    records: list[dict] = []
    seen_words: set[str] = set()
    with (root / "linux_gt.txt").open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            try:
                relative, text = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed linux_gt.txt line {line_no}") from exc
            relative = Path(relative).as_posix()
            word_id = Path(relative).stem
            form_id = form_id_from_word_id(word_id)
            writer_id = form_writers.get(form_id)
            image_path = root / Path(relative)
            if writer_id is None or not image_path.is_file() or not text:
                continue
            if any(char not in alphabet for char in text):
                continue
            if word_id in seen_words:
                raise ValueError(f"Duplicate word id in linux_gt.txt: {word_id}")
            seen_words.add(word_id)
            records.append({
                "word_id": word_id,
                "form_id": form_id,
                "writer_id": writer_id,
                "image": relative,
                "text": text,
            })
    return records


def split_records_by_writer(records: Sequence[dict], seed: int = 42) -> tuple[dict[str, list[dict]], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["writer_id"])].append(dict(record))
    grouped = {writer: items for writer, items in grouped.items()
               if len({item["word_id"] for item in items}) >= 2}
    writers = sorted(grouped)
    random.Random(seed).shuffle(writers)
    n = len(writers)
    if n < 3:
        raise ValueError("At least three writers with two valid images are required")
    n_train = max(1, int(math.floor(n * 0.8)))
    n_val = max(1, int(math.floor(n * 0.1)))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    writer_splits = {
        "train": writers[:n_train],
        "validation": writers[n_train:n_train + n_val],
        "test": writers[n_train + n_val:],
    }
    splits = {
        name: sorted((item for writer in selected for item in grouped[writer]),
                     key=lambda item: item["word_id"])
        for name, selected in writer_splits.items()
    }
    validate_splits(splits)
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "ratios": [0.8, 0.1, 0.1],
        "writer_lists": writer_splits,
        "counts": {name: {"writers": len(writer_splits[name]), "images": len(splits[name])}
                   for name in splits},
    }
    metadata["split_hash"] = _sha256_json({"splits": splits, "writer_lists": writer_splits})
    return splits, metadata


def validate_splits(splits: dict[str, Sequence[dict]]) -> None:
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError(f"Expected splits {sorted(required)}, got {sorted(splits)}")
    for field in ("writer_id", "word_id", "image"):
        values = {name: {str(item[field]) for item in records} for name, records in splits.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = values[left] & values[right]
            if overlap:
                raise ValueError(f"{field} overlap between {left}/{right}: {next(iter(overlap))}")


def write_manifests(root: str | Path, output: str | Path, seed: int = 42) -> dict:
    dataset_root = validate_dataset_root(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits, metadata = split_records_by_writer(read_linux_gt(dataset_root), seed)
    for name, records in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        (output / f"{name}_writers.txt").write_text(
            "\n".join(metadata["writer_lists"][name]) + "\n", encoding="utf-8")
    metadata["dataset_root"] = str(dataset_root)
    (output / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_manifest(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                records.append(json.loads(raw))
    return records


def preprocess_word(image: Image.Image, height: int = 32) -> torch.Tensor:
    image = image.convert("L")
    width = max(1, round(image.width * height / image.height))
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32)
    # HiGAN uses white ink (+1) on a black background (-1).
    return torch.from_numpy((255.0 - array) / 127.5 - 1.0).unsqueeze(0)


class IAMManifestDataset(Dataset):
    def __init__(self, dataset_root: str | Path, manifest: str | Path):
        self.root = validate_dataset_root(dataset_root)
        self.records = load_manifest(manifest)
        writers = sorted({str(item["writer_id"]) for item in self.records})
        self.writer_to_index = {writer: index for index, writer in enumerate(writers)}
        self.converter = strLabelConverter("all")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(self.root / record["image"]) as image:
            tensor = preprocess_word(image)
        return {
            "image": tensor,
            "label": torch.tensor(self.converter.encode(record["text"]), dtype=torch.long),
            "writer": self.writer_to_index[str(record["writer_id"])],
            "writer_id": str(record["writer_id"]),
            "word_id": record["word_id"],
            "text": record["text"],
        }


def collate_words(batch: Sequence[dict]) -> dict:
    image_lens = torch.tensor([item["image"].shape[-1] for item in batch], dtype=torch.int32)
    padded_lens = ((image_lens + 15) // 16) * 16
    max_width = int(padded_lens.max())
    images = torch.full((len(batch), 1, 32, max_width), -1.0)
    for index, item in enumerate(batch):
        images[index, :, :, :item["image"].shape[-1]] = item["image"]
    label_lens = torch.tensor([len(item["label"]) for item in batch], dtype=torch.int32)
    labels = torch.zeros((len(batch), int(label_lens.max())), dtype=torch.long)
    for index, item in enumerate(batch):
        labels[index, :len(item["label"])] = item["label"]
    return {
        "images": images,
        "image_lens": padded_lens,
        "labels": labels,
        "label_lens": label_lens,
        "writers": torch.tensor([item["writer"] for item in batch], dtype=torch.long),
        "writer_ids": [item["writer_id"] for item in batch],
        "word_ids": [item["word_id"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


class StatefulWriterPairBatchSampler(Sampler[list[int]]):
    """Deterministic batches containing two distinct words per unique writer."""

    def __init__(self, records: Sequence[dict], writers_per_batch: int = 8, seed: int = 123456,
                 epoch: int = 0, start_batch: int = 0):
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_batch = int(start_batch)
        self.writers_per_batch = int(writers_per_batch)
        self.by_writer: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(records):
            self.by_writer[str(item["writer_id"])].append(index)
        self.by_writer = {writer: indices for writer, indices in self.by_writer.items()
                          if len({records[index]["word_id"] for index in indices}) >= 2}
        if len(self.by_writer) < self.writers_per_batch:
            raise ValueError(f"Need {self.writers_per_batch} eligible writers, got {len(self.by_writer)}")
        self._batches_per_epoch = max(1, len(records) // (2 * self.writers_per_batch))

    def __len__(self) -> int:
        return max(0, self._batches_per_epoch - self.start_batch)

    def _batch(self, batch_index: int) -> list[int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + batch_index * 9_973)
        writers = rng.sample(sorted(self.by_writer), self.writers_per_batch)
        result = []
        for writer in writers:
            first, second = rng.sample(self.by_writer[writer], 2)
            result.extend((first, second))
        return result

    def __iter__(self) -> Iterator[list[int]]:
        for batch_index in range(self.start_batch, self._batches_per_epoch):
            yield self._batch(batch_index)

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "start_batch": self.start_batch,
                "writers_per_batch": self.writers_per_batch}

    def load_state_dict(self, state: dict) -> None:
        if int(state["seed"]) != self.seed or int(state["writers_per_batch"]) != self.writers_per_batch:
            raise ValueError("Sampler settings do not match checkpoint")
        self.epoch = int(state["epoch"])
        self.start_batch = int(state["start_batch"])

    def set_position(self, epoch: int, next_batch: int = 0) -> None:
        self.epoch, self.start_batch = int(epoch), int(next_batch)
