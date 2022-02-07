import torch
from typing import Optional
import math


def hwc2bchw(images: torch.Tensor) -> torch.Tensor:
    return images.unsqueeze(0).permute(0, 3, 1, 2)


def bchw2hwc(images: torch.Tensor, nrows: Optional[int] = None, border: int = 2,
             background_value: float = 0) -> torch.Tensor:
    """ make a grid image from an image batch.

    Args:
        images (torch.Tensor): input image batch.
        nrows: rows of grid.
        border: border size in pixel.
        background_value: color value of background.
    """
    assert images.ndim == 4  # n x c x h x w
    images = images.permute(0, 2, 3, 1)  # n x h x w x c
    n, h, w, c = images.shape
    if nrows is None:
        nrows = max(int(math.sqrt(n)), 1)
    ncols = (n + nrows - 1) // nrows
    result = torch.full([(h + border) * nrows - border,
                         (w + border) * ncols - border, c], background_value,
                         device=images.device,
                         dtype=images.dtype)

    for i, single_image in enumerate(images):
        row = i // ncols
        col = i % ncols
        yy = (h + border) * row
        xx = (w + border) * col
        result[yy:(yy+h), xx:(xx+w), :] = single_image
    return result
