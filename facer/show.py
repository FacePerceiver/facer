from typing import Optional, Tuple
import torch
from PIL import Image
import matplotlib.pyplot as plt

from .util import bchw2hwc


def set_figsize(size: Optional[Tuple[int, int]] = None):
    if size is None:
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    else:
        plt.rcParams["figure.figsize"] = size


def show_hwc(image: torch.Tensor):
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    pimage = Image.fromarray(image.cpu().numpy())
    plt.imshow(pimage)
    plt.show()


def show_bchw(image: torch.Tensor):
    show_hwc(bchw2hwc(image))
