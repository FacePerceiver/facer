import torch
from PIL import Image
import matplotlib.pyplot as plt

from .util import bchw2hwc


def show_hwc(image: torch.Tensor):
    pimage = Image.fromarray(image.cpu().numpy())
    plt.imshow(pimage)
    plt.show()


def show_bchw(image: torch.Tensor):
    show_hwc(bchw2hwc(image))
