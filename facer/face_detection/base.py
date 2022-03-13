import torch
import torch.nn as nn


class FaceDetector(nn.Module):
    """ face detector

    Args:
        images (torch.Tensor): b x c x h x w

    Returns:
        data (Dict[str, torch.Tensor]):

            * rects: nfaces x 4 (x1, y1, x2, y2)
            * points: nfaces x 5 x 2 (x, y)
            * scores: nfaces
            * image_ids: nfaces
    """
    pass
