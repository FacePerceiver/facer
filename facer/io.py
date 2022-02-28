import torch
import numpy as np
from PIL import Image


def read_hwc(path: str) -> torch.Tensor:
    """Read an image from a given path.

    Args:
        path (str): The given path.
    """
    image = Image.open(path)
    np_image = np.array(image.convert('RGB'))
    return torch.from_numpy(np_image)


def write_hwc(image: torch.Tensor, path: str):
    """Write an image to a given path.

    Args:
        image (torch.Tensor): The image.
        path (str): The given path.
    """

    Image.fromarray(image.cpu().numpy()).save(path)



