import sys
import types

import pytest

from higan_contrastive.metrics import resolve_hwd_checkpoint


def test_hwd_local_checkpoint_is_unchanged():
    assert resolve_hwd_checkpoint("/weights/hwd.pth") == "/weights/hwd.pth"


def test_hwd_hf_uri_resolves_repo_file_and_revision(monkeypatch):
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return "/cache/VGG16.pth"

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=fake_download))
    result = resolve_hwd_checkpoint("hf://owner/hwd/VGG16.pth@v1")
    assert result == "/cache/VGG16.pth"
    assert calls == {"repo_id": "owner/hwd", "filename": "VGG16.pth", "revision": "v1"}


def test_hwd_hf_uri_requires_file():
    with pytest.raises(ValueError):
        resolve_hwd_checkpoint("hf://owner/repo")
