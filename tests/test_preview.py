from PIL import Image
import torch

from higan_contrastive.preview import save_preview


def test_preview_grid_is_written(tmp_path):
    images = torch.full((2, 1, 32, 48), -1.0)
    images[:, :, 8:24, 10:30] = 1.0
    output = tmp_path / "preview.png"
    save_preview(images, torch.tensor([32, 48]), images, images,
                 torch.tensor([32, 48]), ["hello", "world"], output)
    with Image.open(output) as preview:
        assert preview.width > 3 * 32
        assert preview.height > 2 * 32
