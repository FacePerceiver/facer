from typing import Optional
import functools
import torch
import torch.nn.functional as F

from ..util import download_jit
from ..transform import get_face_align_grid, get_face_crop_grid
from .base import FaceParser

pretrain_settings = {
    'celebm/448': {
        'url': [
            'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt',
        ],
        'get_grid_fn': functools.partial(get_face_align_grid, target_shape=(448, 448), target_face_scale=0.8, warp_factor=0.0),
        'label_names': ['background', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                        'le', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                        'glass', 'hat', 'earr', 'neckl']
    },
    'lapa/448': {
        'url': [
            'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit.pt',
        ],
        'get_grid_fn': functools.partial(get_face_align_grid, target_shape=(448, 448), target_face_scale=1.0, warp_factor=0.8),
        'label_names': ['background', 'face', 'rb', 'lb', 're',
                        'le', 'nose',  'ulip', 'imouth', 'llip', 'hair']
    }
}


class FaRLFaceParser(FaceParser):
    def __init__(self, conf_name: Optional[str] = None,
                 model_path: Optional[str] = None) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = 'celebm/448'
        if model_path is None:
            model_path = pretrain_settings[conf_name]['url']
        self.conf_name = conf_name
        self.net = download_jit(model_path)
        self.eval()

    def forward(self, images, data):
        images = images.float() / 255.0
        with torch.no_grad():
            image_ids, grid, inv_grid = pretrain_settings[self.conf_name]['get_grid_fn'](
                images, data)

        w_images = F.grid_sample(
            images[image_ids], grid, mode='bilinear', align_corners=False)

        w_seg_logits, _ = self.net(w_images)  # (b*n) x c x h x w

        seg_logits = F.grid_sample(
            w_seg_logits, inv_grid, mode='bilinear', align_corners=False)

        for image_id, datum in enumerate(data):
            selected = [image_id == i for i in image_ids]
            datum['seg'] = {
                'logits': seg_logits[selected, :, :, :],
                'label_names': pretrain_settings[self.conf_name]['label_names']
            }
        return data
