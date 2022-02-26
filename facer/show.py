from typing import Optional, Tuple
import torch
from PIL import Image
import matplotlib.pyplot as plt

from .util import bchw2hwc


def set_figsize(*args):
    if len(args) == 0:
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    elif len(args) == 1:
        plt.rcParams["figure.figsize"] = (args[0], args[0])
    elif len(args) == 2:
        plt.rcParams["figure.figsize"] = tuple(args)
    else:
        raise RuntimeError(
            f'Supported argument types: set_figsize() or set_figsize(int) or set_figsize(int, int)')


def show_hwc(image: torch.Tensor):
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    if image.size(2) == 1:
        image = image.repeat(1, 1, 3)
    pimage = Image.fromarray(image.cpu().numpy())
    plt.imshow(pimage)
    plt.show()


def show_bchw(image: torch.Tensor):
    show_hwc(bchw2hwc(image))


def show_bhw(image: torch.Tensor):
    show_bchw(image.unsqueeze(1))
