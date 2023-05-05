# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple, Union

import torch


def normalize_points(points: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """ Normalize coordinates to [0, 1].
    """
    return (points + 0.5) / torch.tensor([[[w, h]]]).to(points)


def denormalize_points(normalized_points: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """ Reverse normalize_points.
    """
    return normalized_points * torch.tensor([[[w, h]]]).to(normalized_points) - 0.5


def heatmap2points(heatmap, t_scale: Union[None, float, torch.Tensor] = None):
    """ Heatmaps -> normalized points [b x npoints x 2(XY)].
    """
    dtype = heatmap.dtype
    _, _, h, w = heatmap.shape

    # 0 ~ h-1, 0 ~ w-1
    yy, xx = torch.meshgrid(
        torch.arange(h).float(),
        torch.arange(w).float())

    yy = yy.view(1, 1, h, w).to(heatmap)
    xx = xx.view(1, 1, h, w).to(heatmap)

    if t_scale is not None:
        heatmap = (heatmap * t_scale).exp()
    heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

    yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum  # b x npoints
    xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum  # b x npoints

    points = torch.stack([xx_coord, yy_coord], dim=-1)  # b x npoints x 2

    normalized_points = normalize_points(points, h, w)
    return normalized_points
