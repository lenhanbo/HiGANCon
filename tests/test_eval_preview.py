from pathlib import Path

import torch
from PIL import Image

from higan_contrastive.eval_preview import save_evaluation_previews
from evaluate import encode_query_texts
from scripts.download_hf_checkpoint import infer_repo_type, parse_source


def test_parse_huggingface_checkpoint_sources():
    assert parse_source(
        "https://huggingface.co/alice/higan/resolve/main/checkpoints/latest.pt",
        "ignored.pt", "ignored") == (
            "alice/higan", "checkpoints/latest.pt", "main")
    assert parse_source(
        "https://huggingface.co/alice/higan/blob/v2/latest.pt",
        "ignored.pt", "ignored") == ("alice/higan", "latest.pt", "v2")
    assert parse_source("hf://alice/higan/last.pt@dev", "ignored.pt", "main") == (
        "alice/higan", "last.pt", "dev")
    assert parse_source("alice/higan", "latest.pt", "main") == (
        "alice/higan", "latest.pt", "main")
    assert parse_source(
        "https://huggingface.co/datasets/lenhanbo/higancon/blob/main/higancon.pt",
        "ignored.pt", "ignored") == ("lenhanbo/higancon", "higancon.pt", "main")
    assert infer_repo_type(
        "https://huggingface.co/datasets/lenhanbo/higancon/blob/main/higancon.pt"
    ) == "dataset"
    assert parse_source(
        "https://huggingface.co/datasets/lenhanbo/higancon",
        "latest.pt", "main") == ("lenhanbo/higancon", "latest.pt", "main")
    assert parse_source(
        "datasets/lenhanbo/higancon", "latest.pt", "main"
    ) == ("lenhanbo/higancon", "latest.pt", "main")


def test_evaluation_gallery_contains_every_requested_sample(tmp_path: Path):
    samples = []
    for index in range(3):
        samples.append({
            "sample_index": index,
            "writer_id": f"writer-{index}",
            "word_id": f"word-{index}",
            "query": f"query {index}",
            "reference": torch.full((1, 32, 40 + index), -1.0),
            "generated": torch.full((1, 32, 48 + index), 1.0),
        })
    pages = save_evaluation_previews(samples, tmp_path, page_size=2)
    assert [path.name for path in pages] == ["page_0001.png", "page_0002.png"]
    assert len((tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    with Image.open(pages[0]) as page:
        assert page.width > 900


def test_encode_gallery_queries():
    labels, lengths = encode_query_texts(["that", "if"], torch.device("cpu"))
    assert labels.shape == (2, 4)
    assert lengths.tolist() == [4, 2]
