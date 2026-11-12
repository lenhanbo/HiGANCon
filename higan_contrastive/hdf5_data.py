from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.alphabet import strLabelConverter


HIGAN_HDF5_FILES = {
    "train": "trnvalset_words32.hdf5",
    "test": "testset_words32.hdf5",
    "test_original": "testset_words32_OrgSz.hdf5",
}
HIGAN_HDF5_ALIASES = {
    "test_original": ("testset_words32_OrgSz.hdf5", "testset_words32d_OrgSz.hdf5"),
}
HDF5_KEYS = ("imgs", "lbs", "img_seek_idxs", "lb_seek_idxs", "img_lens", "lb_lens", "wids")


def higan_hdf5_path(root: str | Path, key: str) -> Path:
    root = Path(root)
    candidates = HIGAN_HDF5_ALIASES.get(key, (HIGAN_HDF5_FILES[key],))
    for filename in candidates:
        path = root / filename
        if path.is_file():
            return path
    return root / HIGAN_HDF5_FILES[key]


def validate_higan_hdf5_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    missing = [" or ".join(HIGAN_HDF5_ALIASES.get(key, (filename,)))
               for key, filename in HIGAN_HDF5_FILES.items()
               if not higan_hdf5_path(root, key).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid HiGAN HDF5 root {root}: missing {', '.join(missing)}")
    for key, filename in HIGAN_HDF5_FILES.items():
        path = higan_hdf5_path(root, key)
        with h5py.File(path, "r") as handle:
            missing_keys = [key for key in HDF5_KEYS if key not in handle]
            if missing_keys:
                raise ValueError(f"{path.name} is missing HDF5 keys: {missing_keys}")
    return root


def _indexed_fingerprint(path: Path, digest: Any) -> dict[str, int]:
    """Hash split identity without hashing the large concatenated image byte array."""
    with h5py.File(path, "r") as handle:
        imgs = handle["imgs"]
        digest.update(str((imgs.shape, str(imgs.dtype))).encode())
        for key in ("lbs", "img_seek_idxs", "lb_seek_idxs", "img_lens", "lb_lens", "wids"):
            array = np.asarray(handle[key])
            digest.update(key.encode())
            digest.update(str((array.shape, str(array.dtype))).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return {"images": int(len(handle["img_lens"])),
                "writers": int(len(np.unique(np.asarray(handle["wids"]))))}


def prepare_higan_hdf5_metadata(root: str | Path, output: str | Path) -> dict:
    root = validate_higan_hdf5_root(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    counts = {}
    for name in HIGAN_HDF5_FILES:
        digest.update(name.encode())
        counts[name] = _indexed_fingerprint(higan_hdf5_path(root, name), digest)
    if counts["test"]["images"] != counts["test_original"]["images"]:
        raise ValueError("HiGAN test and test_original HDF5 sample counts differ")
    metadata = {
        "schema_version": 1,
        "dataset_format": "higan_hdf5",
        "hdf5_files": HIGAN_HDF5_FILES,
        "counts": counts,
        "split_hash": digest.hexdigest(),
    }
    (output / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


class HiGANHDF5Dataset(Dataset):
    """Lazy-reader for the exact flattened HDF5 layout distributed with HiGAN."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.converter = strLabelConverter("all")
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            missing = [key for key in HDF5_KEYS if key not in handle]
            if missing:
                raise ValueError(f"{self.path.name} is missing HDF5 keys: {missing}")
            self.img_seek_idxs = np.asarray(handle["img_seek_idxs"], dtype=np.int64)
            self.lb_seek_idxs = np.asarray(handle["lb_seek_idxs"], dtype=np.int64)
            self.img_lens = np.asarray(handle["img_lens"], dtype=np.int64)
            self.lb_lens = np.asarray(handle["lb_lens"], dtype=np.int64)
            self.wids = np.asarray(handle["wids"], dtype=np.int64)
            self.lbs = np.asarray(handle["lbs"])
        n = len(self.img_lens)
        arrays = (self.img_seek_idxs, self.lb_seek_idxs, self.lb_lens, self.wids)
        if any(len(array) != n for array in arrays):
            raise ValueError(f"Inconsistent HDF5 index lengths in {self.path}")
        raw_writers = sorted({int(writer) for writer in self.wids.tolist()})
        self.writer_to_index = {str(writer): index for index, writer in enumerate(raw_writers)}
        self.records = []
        for index in range(n):
            text = self._text(index)
            writer_id = str(int(self.wids[index]))
            self.records.append({
                "hdf5_index": index,
                "word_id": f"{self.path.stem}:{index:08d}",
                "writer_id": writer_id,
                "text": text,
            })

    def _text(self, index: int) -> str:
        start, length = int(self.lb_seek_idxs[index]), int(self.lb_lens[index])
        return "".join(chr(int(value)) for value in self.lbs[start:start + length])

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        start, width = int(self.img_seek_idxs[index]), int(self.img_lens[index])
        pixels = np.asarray(self._file()["imgs"][:, start:start + width], dtype=np.float32).copy()
        if pixels.shape[0] != 32:
            raise ValueError(f"HiGAN HDF5 image height must be 32, got {pixels.shape[0]}")
        image = torch.from_numpy(pixels / 127.5 - 1.0).unsqueeze(0)
        record = self.records[index]
        label = torch.tensor(self.converter.encode(record["text"]), dtype=torch.long)
        return {
            "image": image,
            "label": label,
            "writer": self.writer_to_index[record["writer_id"]],
            "writer_id": record["writer_id"],
            "word_id": record["word_id"],
            "text": record["text"],
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        if hasattr(self, "_handle"):
            self.close()


class AlignedHiGANHDF5EvalDataset(Dataset):
    """Style references from resized test HDF5, metric reals from _OrgSz HDF5."""

    def __init__(self, reference_path: str | Path, original_path: str | Path):
        self.reference = HiGANHDF5Dataset(reference_path)
        self.original = HiGANHDF5Dataset(original_path)
        if len(self.reference) != len(self.original):
            raise ValueError("HiGAN reference/original test HDF5 lengths differ")
        for index, (left, right) in enumerate(zip(self.reference.records, self.original.records)):
            if left["text"] != right["text"] or left["writer_id"] != right["writer_id"]:
                raise ValueError(f"HiGAN test HDF5 files are not aligned at index {index}")
        self.records = self.reference.records
        self.writer_to_index = self.reference.writer_to_index

    def __len__(self) -> int:
        return len(self.reference)

    def __getitem__(self, index: int) -> dict:
        item = self.reference[index]
        original = self.original[index]
        item["metric_real_image"] = original["image"]
        return item


def load_higan_hdf5_dataset(root: str | Path, split: str) -> Dataset:
    root = validate_higan_hdf5_root(root)
    if split == "train":
        return HiGANHDF5Dataset(higan_hdf5_path(root, "train"))
    if split == "test":
        return AlignedHiGANHDF5EvalDataset(
            higan_hdf5_path(root, "test"), higan_hdf5_path(root, "test_original"))
    raise ValueError(f"Unsupported HiGAN HDF5 split: {split}")
