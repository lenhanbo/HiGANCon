from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from .preview import _display_image


def save_alignment_heatmap(weights: torch.Tensor, text: str, path: str | Path,
                           entropy: torch.Tensor | None = None,
                           peaks: torch.Tensor | None = None,
                           title: str = "CTC ALIGNMENT") -> Path:
    """Write a dependency-free occurrence-by-timestep posterior heatmap."""
    count = min(len(text), weights.shape[0])
    matrix = weights[:count].detach().float().cpu().numpy()
    if count == 0:
        matrix = np.zeros((1, max(1, weights.shape[-1])), dtype=np.float32)
    maximum = float(matrix.max()) if matrix.size else 0.0
    normalized = matrix / maximum if maximum > 0 else matrix
    cell_x, cell_y, left, top = 4, 18, 150, 28
    width = left + normalized.shape[1] * cell_x + 8
    height = top + normalized.shape[0] * cell_y + 8
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill="black")
    entropy_values = entropy.detach().cpu().tolist() if entropy is not None else []
    peak_values = peaks.detach().cpu().tolist() if peaks is not None else []
    for row in range(normalized.shape[0]):
        character = text[row] if row < len(text) else "?"
        peak = int(peak_values[row]) if row < len(peak_values) else int(matrix[row].argmax())
        ent = float(entropy_values[row]) if row < len(entropy_values) else 0.0
        draw.text((6, top + row * cell_y + 2),
                  f"{row}:{character!r} peak={peak} H={ent:.3f}", fill="black")
        for column, value in enumerate(normalized[row]):
            color = (255, int(255 * (1.0 - value)), int(255 * (1.0 - value)))
            draw.rectangle((left + column * cell_x, top + row * cell_y,
                            left + (column + 1) * cell_x - 1,
                            top + (row + 1) * cell_y - 2), fill=color)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def save_ctc_preview(style_reference: torch.Tensor, style_lengths: torch.Tensor,
                     real_target: torch.Tensor, real_lengths: torch.Tensor,
                     paired_generated: torch.Tensor, paired_lengths: torch.Tensor,
                     random_generated: torch.Tensor, random_lengths: torch.Tensor,
                     reference_texts: Sequence[str], query_texts: Sequence[str],
                     path: str | Path, max_items: int = 4,
                     writer_ids: Sequence[str] | None = None,
                     supcon_images: torch.Tensor | None = None,
                     supcon_lengths: torch.Tensor | None = None,
                     supcon_text: str | None = None, metadata: dict | None = None) -> Path:
    """Save the complete style/query/target preview grid for soft matching."""
    count = min(max_items, len(real_target), len(query_texts))
    if count <= 0:
        raise ValueError("Preview batch is empty")
    tensors = [style_reference, real_target, paired_generated, random_generated]
    lengths = [style_lengths, real_lengths, paired_lengths, random_lengths]
    titles = ["STYLE REFERENCE", "REAL TARGET", "PAIRED GENERATED", "RANDOM GENERATED"]
    if supcon_images is not None:
        if supcon_lengths is None:
            raise ValueError("supcon_lengths is required with supcon_images")
        tensors.append(supcon_images)
        lengths.append(supcon_lengths)
        titles.append("SUPCON SHARED QUERY")
    groups = [[_display_image(tensor[i], int(size[i])) for i in range(count)]
              for tensor, size in zip(tensors, lengths)]
    widths = [max(max(tile.width for tile in group), len(title) * 7)
              for group, title in zip(groups, titles)]
    margin, gap, header, label_height, image_height = 8, 8, 22, 30, 32
    canvas = Image.new("L", (2 * margin + sum(widths) + gap * (len(groups) - 1),
                             margin + header + count * (image_height + label_height + gap)), 255)
    draw = ImageDraw.Draw(canvas)
    positions, current = [], margin
    for width, title in zip(widths, titles):
        positions.append(current)
        draw.text((current, margin), title, fill=0)
        current += width + gap
    for row in range(count):
        y = margin + header + row * (image_height + label_height + gap)
        for column, group in enumerate(groups):
            canvas.paste(group[row], (positions[column], y))
        writer = writer_ids[row] if writer_ids is not None else "?"
        label = (f"writer={writer} | ref={reference_texts[row]} | query={query_texts[row]}"
                 + (f" | supcon={supcon_text}" if supcon_text is not None else ""))
        draw.text((margin, y + image_height + 1), label[:180], fill=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    if metadata is not None:
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
