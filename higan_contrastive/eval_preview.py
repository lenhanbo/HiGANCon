from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


def _word_image(tensor: torch.Tensor, panel_width: int = 320,
                panel_height: int = 64) -> Image.Image:
    """Render an internal HiGAN tensor as black ink on a fixed white panel."""
    pixels = tensor.detach().float().cpu()[0].clamp(-1, 1)
    array = ((1.0 - (pixels + 1.0) / 2.0) * 255.0).round().byte().numpy()
    word = Image.fromarray(np.ascontiguousarray(array), mode="L")
    scale = min(panel_width / max(1, word.width), panel_height / max(1, word.height))
    resized = word.resize((max(1, round(word.width * scale)),
                           max(1, round(word.height * scale))), Image.Resampling.BILINEAR)
    panel = Image.new("L", (panel_width, panel_height), 255)
    panel.paste(resized, (0, (panel_height - resized.height) // 2))
    return panel


def save_evaluation_previews(samples: Sequence[dict], output: str | Path,
                             page_size: int = 16) -> list[Path]:
    """Save paged REFERENCE | QUERY TEXT | GENERATED galleries and JSONL metadata."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not samples:
        return []

    metadata_path = output / "samples.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps({key: value for key, value in sample.items()
                                     if key not in {"reference", "generated"}},
                                    ensure_ascii=False, sort_keys=True) + "\n")

    panel_width, panel_height = 320, 64
    gap, margin, header_height, label_height = 12, 10, 25, 22
    row_height = panel_height + label_height + gap
    page_paths = []
    for page_index, start in enumerate(range(0, len(samples), page_size), 1):
        page = samples[start:start + page_size]
        width = margin * 2 + panel_width * 3 + gap * 2
        height = margin * 2 + header_height + len(page) * row_height
        canvas = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(canvas)
        positions = [margin, margin + panel_width + gap,
                     margin + (panel_width + gap) * 2]
        for x, title in zip(positions, ("REFERENCE", "QUERY TEXT", "GENERATED")):
            draw.text((x, margin), title, fill=0)

        for row, sample in enumerate(page):
            y = margin + header_height + row * row_height
            canvas.paste(_word_image(sample["reference"], panel_width, panel_height),
                         (positions[0], y))
            query_panel = Image.new("L", (panel_width, panel_height), 255)
            query_draw = ImageDraw.Draw(query_panel)
            query_lines = textwrap.wrap(str(sample["query"]), width=42) or [""]
            query_draw.multiline_text((5, 8), "\n".join(query_lines[:3]), fill=0, spacing=3)
            canvas.paste(query_panel, (positions[1], y))
            canvas.paste(_word_image(sample["generated"], panel_width, panel_height),
                         (positions[2], y))
            label = (f"sample={sample['sample_index']} | writer={sample['writer_id']} | "
                     f"word_id={sample['word_id']} | ref_text={sample.get('reference_text', '')}")
            draw.text((margin, y + panel_height + 2), label[:180], fill=0)

        page_path = output / f"page_{page_index:04d}.png"
        canvas.save(page_path)
        page_paths.append(page_path)
    return page_paths
