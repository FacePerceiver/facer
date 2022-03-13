from numpy import isin
import torch
from typing import Any, Generator, Optional, Tuple, Union, List, Dict
import math
import os
from urllib.parse import urlparse
import errno
import sys
import validators


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


def select_data(selection, data):
    if isinstance(data, dict):
        return {name: select_data(selection, val) for name, val in data.items()}
    elif isinstance(data, (list, tuple)):
        return [select_data(selection, val) for val in data]
    elif isinstance(data, torch.Tensor):
        return data[selection]
    return data


def download_jit(url_or_paths: Union[str, List[str]], model_dir=None, map_location=None):
    if isinstance(url_or_paths, str):
        url_or_paths = [url_or_paths]

    for url_or_path in url_or_paths:
        try:
            if validators.url(url_or_path):
                url = url_or_path
                if model_dir is None:
                    hub_dir = torch.hub.get_dir()
                    model_dir = os.path.join(hub_dir, 'checkpoints')

                try:
                    os.makedirs(model_dir)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        # Directory already exists, ignore.
                        pass
                    else:
                        # Unexpected OSError, re-raise.
                        raise

                parts = urlparse(url)
                filename = os.path.basename(parts.path)
                cached_file = os.path.join(model_dir, filename)
                if not os.path.exists(cached_file):
                    sys.stderr.write(
                        'Downloading: "{}" to {}\n'.format(url, cached_file))
                    hash_prefix = None
                    torch.hub.download_url_to_file(
                        url, cached_file, hash_prefix, progress=True)
            else:
                cached_file = url_or_path

            return torch.jit.load(cached_file, map_location=map_location)
        except:
            sys.stderr.write(f'failed downloading from {url_or_path}\n')

    raise RuntimeError('failed to download jit models from all given urls')
