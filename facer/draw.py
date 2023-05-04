from typing import Dict, List
import torch
import colorsys
import random
import numpy as np
from skimage.draw import line_aa, circle_perimeter_aa
import cv2
from .util import select_data


def _gen_random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


_static_label_colors = [
    np.array((1.0, 1.0, 1.0), np.float32),
    np.array((255, 250, 79), np.float32) / 255.0,  # face
    np.array([255, 125, 138], np.float32) / 255.0,  # lb
    np.array([213, 32, 29], np.float32) / 255.0,  # rb
    np.array([0, 144, 187], np.float32) / 255.0,  # le
    np.array([0, 196, 253], np.float32) / 255.0,  # re
    np.array([255, 129, 54], np.float32) / 255.0,  # nose
    np.array([88, 233, 135], np.float32) / 255.0,  # ulip
    np.array([0, 117, 27], np.float32) / 255.0,  # llip
    np.array([255, 76, 249], np.float32) / 255.0,  # imouth
    np.array((1.0, 0.0, 0.0), np.float32),  # hair
    np.array((255, 250, 100), np.float32) / 255.0,  # lr
    np.array((255, 250, 100), np.float32) / 255.0,  # rr
    np.array((250, 245, 50), np.float32) / 255.0,  # neck
    np.array((0.0, 1.0, 0.5), np.float32),  # cloth
    np.array((1.0, 0.0, 0.5), np.float32),
] + _gen_random_colors(256)

_names_in_static_label_colors = [
    'background', 'face', 'lb', 'rb', 'le', 're', 'nose',
    'ulip', 'llip', 'imouth', 'hair', 'lr', 'rr', 'neck',
    'cloth', 'eyeg', 'hat', 'earr'
]


def _blend_labels(image, labels, label_names_dict=None,
                  default_alpha=0.6, color_offset=None):
    assert labels.ndim == 2
    bg_mask = labels == 0
    if label_names_dict is None:
        colors = _static_label_colors
    else:
        colors = [np.array((1.0, 1.0, 1.0), np.float32)]
        for i in range(1, labels.max() + 1):
            if isinstance(label_names_dict, dict) and i not in label_names_dict:
                bg_mask = np.logical_or(bg_mask, labels == i)
                colors.append(np.zeros((3)))
                continue
            label_name = label_names_dict[i]
            if label_name in _names_in_static_label_colors:
                color = _static_label_colors[
                    _names_in_static_label_colors.index(
                        label_name)]
            else:
                color = np.array((1.0, 1.0, 1.0), np.float32)
            colors.append(color)

    if color_offset is not None:
        ncolors = []
        for c in colors:
            nc = np.array(c)
            if (nc != np.zeros(3)).any():
                nc += color_offset
            ncolors.append(nc)
        colors = ncolors

    if image is None:
        image = orig_image = np.zeros(
            [labels.shape[0], labels.shape[1], 3], np.float32)
        alpha = 1.0
    else:
        orig_image = image / np.max(image)
        image = orig_image * (1.0 - default_alpha)
        alpha = default_alpha
    for i in range(1, np.max(labels) + 1):
        image += alpha * \
            np.tile(
                np.expand_dims(
                    (labels == i).astype(np.float32), -1),
                [1, 1, 3]) * colors[(i) % len(colors)]
    image[np.where(image > 1.0)] = 1.0
    image[np.where(image < 0)] = 0.0
    image[np.where(bg_mask)] = orig_image[np.where(bg_mask)]
    return image


def _draw_hwc(image: torch.Tensor, data: Dict[str, torch.Tensor]):
    device = image.device
    image = np.array(image.cpu().numpy(), copy=True)
    dtype = image.dtype
    h, w, _ = image.shape

    draw_score_error = False
    for tag, batch_content in data.items():
        if tag == 'rects':
            for cid, content in enumerate(batch_content):
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

                if 'scores' in data:
                    try:
                        import cv2
                        score = data['scores'][cid].item()
                        score_str = f'{score:0.3f}'
                        image_c = np.array(image).copy()
                        cv2.putText(image_c, score_str, org=(int(x1), int(y2)),
                                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                    fontScale=0.6, color=(255, 255, 255), thickness=1)
                        image[:, :, :] = image_c
                    except Exception as e:
                        if not draw_score_error:
                            print(f'Failed to draw scores on image.')
                            print(e)
                        draw_score_error = True

        if tag == 'points':
            for content in batch_content:
                # content: npoints x 2
                for x, y in content:
                    x = max(min(int(x), w-1), 0)
                    y = max(min(int(y), h-1), 0)
                    rr, cc, val = circle_perimeter_aa(y, x, 1)
                    valid = np.all([rr >= 0, rr < h, cc >= 0, cc < w], axis=0)
                    rr = rr[valid]
                    cc = cc[valid]
                    val = val[valid]
                    val = val[:, None][:, [0, 0, 0]]
                    image[rr, cc] = image[rr, cc] * (1.0-val) + val * 255

        if tag == 'seg':
            label_names = batch_content['label_names']
            for seg_logits in batch_content['logits']:
                # content: nclasses x h x w
                seg_probs = seg_logits.softmax(dim=0)
                seg_labels = seg_probs.argmax(dim=0).cpu().numpy()
                image = (_blend_labels(image.astype(np.float32) /
                         255, seg_labels,
                         label_names_dict=label_names) * 255).astype(dtype)

    return torch.from_numpy(image).to(device=device)


def draw_bchw(images: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
    images2 = []
    for image_id, image_chw in enumerate(images):
        selected_data = select_data(image_id == data['image_ids'], data)
        images2.append(
            _draw_hwc(image_chw.permute(1, 2, 0), selected_data).permute(2, 0, 1))
    return torch.stack(images2, dim=0)

def draw_landmarks(img, bbox=None, landmark=None, color=(0, 255, 0)):
    """
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    """
    img = cv2.UMat(img).get()
    if bbox is not None:
        x1, y1, x2, y2 = np.array(bbox)[:4].astype(np.int32)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if landmark is not None:
        for x, y in np.array(landmark).astype(np.int32):
            cv2.circle(img, (int(x), int(y)), 2, color, -1)
    return img