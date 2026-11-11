import json
from pathlib import Path

import pytest
from PIL import Image

from higan_contrastive.data import (StatefulWriterPairBatchSampler, form_id_from_word_id,
                                    parse_forms, validate_dataset_root, write_manifests)


def _iam_tree(tmp_path: Path) -> Path:
    root = tmp_path / "iam_dataset"
    (root / "words" / "a01" / "a01-000u").mkdir(parents=True)
    (root / "forms.txt").write_text("a01-000u 000 2 all 1 1 1 1\n"
                                    "a01-001u 001 2 all 1 1 1 1\n"
                                    "a01-002u 002 2 all 1 1 1 1\n", encoding="utf-8")
    lines = []
    for writer, form in enumerate(("a01-000u", "a01-001u", "a01-002u")):
        target = root / "words" / "a01" / form
        target.mkdir(parents=True, exist_ok=True)
        for word in range(2):
            word_id = f"{form}-00-0{word}"
            relative = f"words/a01/{form}/{word_id}.png"
            Image.new("L", (20 + word, 12), 255).save(root / relative)
            lines.append(f"{relative}\tword{writer}{word}")
    (root / "linux_gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "train_gt.txt").write_text("compat\n", encoding="utf-8")
    (root / "val_gt.txt").write_text("compat\n", encoding="utf-8")
    return root


def test_form_mapping_and_writer_disjoint_split(tmp_path):
    root = _iam_tree(tmp_path)
    assert parse_forms(root / "forms.txt")["a01-000u"] == "000"
    assert form_id_from_word_id("a01-000u-00-00") == "a01-000u"
    output = tmp_path / "manifests"
    metadata = write_manifests(root, output, seed=42)
    writers = [set((output / f"{name}_writers.txt").read_text().split())
               for name in ("train", "validation", "test")]
    assert not (writers[0] & writers[1] or writers[0] & writers[2] or writers[1] & writers[2])
    assert len(metadata["split_hash"]) == 64


def test_dataset_root_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset_root(tmp_path)


def test_pair_sampler_unique_writers_and_words():
    records = [{"writer_id": f"w{writer}", "word_id": f"w{writer}-{word}"}
               for writer in range(4) for word in range(3)]
    sampler = StatefulWriterPairBatchSampler(records, writers_per_batch=4, seed=7)
    batch = next(iter(sampler))
    pairs = [batch[index:index + 2] for index in range(0, len(batch), 2)]
    assert len({records[pair[0]]["writer_id"] for pair in pairs}) == 4
    assert all(records[a]["word_id"] != records[b]["word_id"] for a, b in pairs)
    assert batch == next(iter(StatefulWriterPairBatchSampler(records, 4, 7)))
