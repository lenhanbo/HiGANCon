from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
                 path: str | Path, max_items: int = 4) -> Path:
    count = min(max_items, len(real), len(texts))
    if count <= 0:
        raise ValueError("Preview batch is empty")
    real_tiles = [_display_image(real[i], int(real_lengths[i])) for i in range(count)]
    random_tiles = [_display_image(random_style[i], int(generated_lengths[i])) for i in range(count)]
    reference_tiles = [_display_image(reference_style[i], int(generated_lengths[i])) for i in range(count)]
    groups = (real_tiles, random_tiles, reference_tiles)
    column_widths = [max(image.width for image in group) for group in groups]
    margin, gap, header_height, label_height = 8, 8, 22, 18
    row_height = 32 + label_height + gap
    canvas_width = margin * 2 + sum(column_widths) + gap * 2
    canvas_height = margin + header_height + count * row_height
    canvas = Image.new("L", (canvas_width, canvas_height), color=255)
    draw = ImageDraw.Draw(canvas)
    x_positions = [margin, margin + column_widths[0] + gap,
                   margin + column_widths[0] + gap + column_widths[1] + gap]
    for x, title in zip(x_positions, ("REAL", "RANDOM STYLE", "REFERENCE STYLE")):
        draw.text((x, margin), title, fill=0)
    for row in range(count):
        y = margin + header_height + row * row_height
        for column, group in enumerate(groups):
            canvas.paste(group[row], (x_positions[column], y))
        label = str(texts[row]).replace("\n", " ")[:80]
        draw.text((margin, y + 34), label, fill=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path
