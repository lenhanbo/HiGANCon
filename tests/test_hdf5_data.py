from pathlib import Path

import h5py
import numpy as np

from higan_contrastive.data import StatefulWriterPairBatchSampler, collate_words
from higan_contrastive.hdf5_data import (HIGAN_HDF5_FILES, HiGANHDF5Dataset,
                                         load_higan_hdf5_dataset,
                                         prepare_higan_hdf5_metadata)


def _write_higan_file(path: Path, widths: list[int]) -> None:
    texts = ["one", "two", "red", "blue"]
    writers = np.array([7, 7, 9, 9], dtype=np.int64)
    images = [np.full((32, width), index * 50, dtype=np.uint8)
              for index, width in enumerate(widths)]
    img_lens = np.array(widths, dtype=np.int64)
    img_seek_idxs = np.concatenate(([0], np.cumsum(img_lens)[:-1])).astype(np.int64)
    encoded = [np.array([ord(char) for char in text], dtype=np.int32) for text in texts]
    lb_lens = np.array([len(item) for item in encoded], dtype=np.int64)
    lb_seek_idxs = np.concatenate(([0], np.cumsum(lb_lens)[:-1])).astype(np.int64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("imgs", data=np.concatenate(images, axis=1))
        handle.create_dataset("lbs", data=np.concatenate(encoded))
        handle.create_dataset("img_seek_idxs", data=img_seek_idxs)
        handle.create_dataset("lb_seek_idxs", data=lb_seek_idxs)
        handle.create_dataset("img_lens", data=img_lens)
        handle.create_dataset("lb_lens", data=lb_lens)
        handle.create_dataset("wids", data=writers)


def _hdf5_root(tmp_path: Path) -> Path:
    root = tmp_path / "hdf5"
    root.mkdir()
    for key, filename in HIGAN_HDF5_FILES.items():
        widths = [25, 31, 38, 47] if "original" not in key else [27, 35, 43, 51]
        _write_higan_file(root / filename, widths)
    return root


def test_higan_hdf5_metadata_and_lazy_dataset(tmp_path):
    root = _hdf5_root(tmp_path)
    metadata = prepare_higan_hdf5_metadata(root, tmp_path / "metadata")
    assert metadata["dataset_format"] == "higan_hdf5"
    assert len(metadata["split_hash"]) == 64
    dataset = HiGANHDF5Dataset(root / HIGAN_HDF5_FILES["train"])
    item = dataset[1]
    assert item["text"] == "two"
    assert item["writer_id"] == "7"
    assert item["image"].shape == (1, 32, 31)
    assert len(dataset.records) == 4
    sampler = StatefulWriterPairBatchSampler(dataset.records, writers_per_batch=2, seed=7)
    train_batch = collate_words([dataset[index] for index in next(iter(sampler))])
    assert train_batch["images"].shape[0] == 4


def test_higan_eval_pairs_reference_with_original_real(tmp_path):
    root = _hdf5_root(tmp_path)
    dataset = load_higan_hdf5_dataset(root, "test")
    batch = collate_words([dataset[0], dataset[1]])
    assert batch["images"].shape[-1] == 32
    assert batch["metric_real_images"].shape[-1] == 48
    assert batch["texts"] == ["one", "two"]
    assert batch["writer_ids"] == ["7", "7"]


def test_higan_accepts_original_readme_orgsz_typo(tmp_path):
    root = _hdf5_root(tmp_path)
    canonical = root / HIGAN_HDF5_FILES["test_original"]
    canonical.rename(root / "testset_words32d_OrgSz.hdf5")
    dataset = load_higan_hdf5_dataset(root, "test")
    assert len(dataset) == 4
