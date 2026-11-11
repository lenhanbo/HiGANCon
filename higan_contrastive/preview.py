from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import numpy as np
import torch
from PIL import Image, ImageDraw


def _display_image(tensor: torch.Tensor, width: int) -> Image.Image:
    """Convert HiGAN's white-on-black [-1,1] tensor to black ink on white."""
    tensor = tensor.detach().float().cpu()[0, :, :max(1, width)].clamp(-1, 1)
    array = ((1.0 - (tensor + 1.0) / 2.0) * 255.0).round().byte().numpy()
    return Image.fromarray(np.ascontiguousarray(array), mode="L")


def save_preview(real: torch.Tensor, real_lengths: torch.Tensor,
                 random_style: torch.Tensor, reference_style: torch.Tensor,
                 generated_lengths: torch.Tensor, texts: Sequence[str],
                 path: str | Path, max_items: int = 4,
                 writer_ids: Sequence[str] | None = None,
                 random_title: str = "RANDOM SYNTHETIC STYLE",
                 contrastive_style: torch.Tensor | None = None,
                 contrastive_lengths: torch.Tensor | None = None,
                 contrastive_text: str | None = None,
                 metadata: dict | None = None) -> Path:
    count = min(max_items, len(real), len(texts))
    if count <= 0:
        raise ValueError("Preview batch is empty")
    real_tiles = [_display_image(real[i], int(real_lengths[i])) for i in range(count)]
    random_tiles = [_display_image(random_style[i], int(generated_lengths[i])) for i in range(count)]
    reference_tiles = [_display_image(reference_style[i], int(generated_lengths[i])) for i in range(count)]
    groups = [real_tiles, random_tiles, reference_tiles]
    titles = ["REAL REFERENCE", random_title, "REFERENCE-GUIDED"]
    if contrastive_style is not None:
        if contrastive_lengths is None:
            raise ValueError("contrastive_lengths is required with contrastive_style")
        groups.append([_display_image(contrastive_style[i], int(contrastive_lengths[i]))
                       for i in range(count)])
        titles.append("SUPCON SHARED TEXT/NOISE")
    column_widths = [max(max(image.width for image in group), len(title) * 7)
                     for group, title in zip(groups, titles)]
    margin, gap, header_height, label_height = 8, 8, 22, 18
    row_height = 32 + label_height + gap
    canvas_width = margin * 2 + sum(column_widths) + gap * (len(groups) - 1)
    canvas_height = margin + header_height + count * row_height
    canvas = Image.new("L", (canvas_width, canvas_height), color=255)
    draw = ImageDraw.Draw(canvas)
    x_positions = []
    current_x = margin
    for width in column_widths:
        x_positions.append(current_x)
        current_x += width + gap
    for x, title in zip(x_positions, titles):
        draw.text((x, margin), title, fill=0)
    for row in range(count):
        y = margin + header_height + row * row_height
        for column, group in enumerate(groups):
            canvas.paste(group[row], (x_positions[column], y))
        writer = f"writer={writer_ids[row]} | " if writer_ids is not None else ""
        shared = f" | supcon_text={contrastive_text}" if contrastive_text is not None else ""
        label = (writer + "text=" + str(texts[row]).replace("\n", " ") + shared)[:140]
        draw.text((margin, y + 34), label, fill=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    if metadata is not None:
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
