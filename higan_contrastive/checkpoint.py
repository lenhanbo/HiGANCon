from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import torch


def state_hash(models: dict[str, torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for model_name in sorted(models):
        for name, tensor in sorted(models[model_name].state_dict().items()):
            digest.update(f"{model_name}.{name}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_config_hash_without_architecture(path: str | Path) -> str | None:
    """Hash config.yaml as it existed before the appended architecture section."""
    data = Path(path).read_bytes()
    marker = b"architecture:"
    index = data.find(marker)
    if index < 0:
        return None
    prefix = data[:index]
    if prefix.endswith(b"\r\n\r\n"):
        prefix = prefix[:-2]
    elif prefix.endswith(b"\n\n"):
        prefix = prefix[:-1]
    return hashlib.sha256(prefix).hexdigest()


def legacy_config_hash_without_evaluation_setup(path: str | Path) -> str | None:
    """Hash config.yaml before the non-training ``evaluation.setup`` key existed."""
    data = Path(path).read_bytes()
    legacy, replacements = re.subn(
        rb"(?m)^  setup:[ \t]*[^\r\n]*(?:\r?\n)", b"", data, count=1)
    return hashlib.sha256(legacy).hexdigest() if replacements else None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
