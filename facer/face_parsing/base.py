import torch
import torch.nn as nn

class FaceParser(nn.Module):
    """ face parser

    Args:
        images (torch.Tensor): b x c x h x w

        data (List[Dict[str, Any]]):

            * rects (torch.Tensor): nfaces x 4 (x1, y1, x2, y2)
            * points (torch.Tensor): nfaces x 5 x 2 (x, y)

    Returns:
        data (List[Dict[str, Any]]):

            * rects (torch.Tensor): nfaces x 4 (x1, y1, x2, y2)
            * points (torch.Tensor): nfaces x 5 x 2 (x, y)
            * seg (Dict[str, Any]):

                * logits (torch.Tensor): nfaces x nclasses x h x w
                * label_names (List[str]): nclasses
    """
    pass