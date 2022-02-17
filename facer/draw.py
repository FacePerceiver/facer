from typing import Dict, List
from cv2 import circle
import torch
from skimage.draw import line_aa, circle_perimeter_aa


def _draw_hwc(image: torch.Tensor, data: Dict[str, torch.Tensor]):
    device = image.device
    image = image.cpu().numpy()
    h, w, _ = image.shape

    for tag, batch_content in data.items():
        if tag == 'rects':
            for content in batch_content:
                x1, y1, x2, y2 = [int(v) for v in content]
                y1, y2 = [max(min(v, h-1), 0) for v in [y1, y2]]
                x1, x2 = [max(min(v, w-1), 0) for v in [x1, x2]]
                for xx1, yy1, xx2, yy2 in [
                    [x1, y1, x2, y1],
                    [x1, y2, x2, y2],
                    [x1, y1, x1, y2],
                    [x2, y1, x2, y2]
                ]:
                    rr, cc, val = line_aa(yy1, xx1, yy2, xx2)
                    val = val[:, None][:, [0, 0, 0]]
                    image[rr, cc] = image[rr, cc] * (1.0-val) + val * 255
        if tag == 'points':
            for content in batch_content:
                # content: npoints x 2
                for x, y in content:
                    x = max(min(int(x), w-1), 0)
                    y = max(min(int(y), h-1), 0)
                    rr, cc, val = circle_perimeter_aa(y, x, 1)
                    val = val[:, None][:, [0, 0, 0]]
                    image[rr, cc] = image[rr, cc] * (1.0-val) + val * 255

    return torch.from_numpy(image).to(device=device)


def draw_bchw(images: torch.Tensor, data: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    images2 = []
    for image_chw, d in zip(images, data):
        images2.append(
            _draw_hwc(image_chw.permute(1, 2, 0), d).permute(2, 0, 1))
    return torch.stack(images2, dim=0)
