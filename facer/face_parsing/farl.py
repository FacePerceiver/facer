from typing import Optional, Dict, Any
import functools
import torch
import torch.nn.functional as F

from ..util import download_jit
from ..transform import (get_crop_and_resize_matrix, get_face_align_matrix, get_face_align_matrix_celebm,
                         make_inverted_tanh_warp_grid, make_tanh_warp_grid)
from .base import FaceParser
import numpy as np

pretrain_settings = {
    'lapa/448': {
        'url': [
            'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt',
        ],
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix,
                                           target_shape=(448, 448), target_face_scale=1.0),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0.8, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0.8, warped_shape=(448, 448)),
        'label_names': ['background', 'face', 'rb', 'lb', 're',
                        'le', 'nose',  'ulip', 'imouth', 'llip', 'hair']
    },
    'celebm/448': {
        'url': [
            'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt',
        ],
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix_celebm,
                                           target_shape=(448, 448)),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0, warped_shape=(448, 448)),
        'label_names':  [
                    'background', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                    'eyeg', 'hat', 'earr', 'neck_l']
    }
}


class FaRLFaceParser(FaceParser):
    """ The face parsing models from [FaRL](https://github.com/FacePerceiver/FaRL).

    Please consider citing 
    ```bibtex
        @article{zheng2021farl,
            title={General Facial Representation Learning in a Visual-Linguistic Manner},
            author={Zheng, Yinglin and Yang, Hao and Zhang, Ting and Bao, Jianmin and Chen, 
                Dongdong and Huang, Yangyu and Yuan, Lu and Chen, 
                Dong and Zeng, Ming and Wen, Fang},
            journal={arXiv preprint arXiv:2112.03109},
            year={2021}
        }
    ```
    """

    def __init__(self, conf_name: Optional[str] = None, model_path: Optional[str] = None, device=None) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = 'lapa/448'
        if model_path is None:
            model_path = pretrain_settings[conf_name]['url']
        self.conf_name = conf_name
        self.net = download_jit(model_path, map_location=device)
        self.eval()
        self.device = device
        self.setting = pretrain_settings[conf_name]
        self.label_names = self.setting['label_names']

    
    def get_warp_grid(self, images: torch.Tensor, matrix_src):
        _, _, h, w = images.shape
        matrix = self.setting['get_matrix_fn'](matrix_src)
        grid = self.setting['get_grid_fn'](matrix=matrix, orig_shape=(h, w))
        inv_grid = self.setting['get_inv_grid_fn'](matrix=matrix, orig_shape=(h, w))
        return grid, inv_grid

    def warp_images(self, images: torch.Tensor, data: Dict[str, Any]):
        simages = self.unify_image_dtype(images)
        simages = simages[data['image_ids']]
        matrix_src = data[self.setting['matrix_src_tag']]
        grid, inv_grid = self.get_warp_grid(simages, matrix_src)

        w_images = F.grid_sample(
            simages, grid, mode='bilinear', align_corners=False)
        return w_images, grid, inv_grid
    

    def decode_image_to_cv2(self, images: torch.Tensor):
        '''
        output: b x 3 x h x w, torch.uint8, [0, 255]
        '''
        assert images.ndim == 4
        assert images.shape[1] == 3
        images = images.permute(0, 2, 3, 1).cpu().numpy() * 255
        images = images.astype(np.uint8)
        return images

    def unify_image_dtype(self, images: torch.Tensor|np.ndarray|list):
        '''
        output: b x 3 x h x w, torch.float32, [0, 1]
        '''
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        elif isinstance(images, torch.Tensor):
            pass
        elif isinstance(images, list):
            assert len(images) > 0, "images is empty"
            first_image = images[0]
            if isinstance(first_image, np.ndarray):
                images = [torch.from_numpy(image).permute(2, 0, 1) for image in images]
                images = torch.stack(images)
            elif isinstance(first_image, torch.Tensor):
                images = torch.stack(images)
            else:
                raise ValueError(f"Unsupported image type: {type(first_image)}")

        else:
            raise ValueError(f"Unsupported image type: {type(images)}")
        
        assert images.ndim == 4
        assert images.shape[1] == 3

        max_val = images.max()
        if max_val <= 1:
            assert images.dtype == torch.float32 or images.dtype == torch.float16
        elif max_val <= 255:
            assert images.dtype == torch.uint8
            images = images.float() / 255.0
        else:
            raise ValueError(f"Unsupported image type: {images.dtype}")
        if images.device != self.device:
            images = images.to(device=self.device)
        return images
    
    @torch.no_grad()
    @torch.inference_mode()
    def forward(self, images: torch.Tensor, data: Dict[str, Any]):
        '''
        images: b x 3 x h x w , torch.uint8, [0, 255]
        data: {'rects': rects, 'points': points, 'scores': scores, 'image_ids': image_ids}
        '''
        w_images, grid, inv_grid = self.warp_images(images, data)
        w_seg_logits = self.forward_warped(w_images, return_preds=False)

        seg_logits = F.grid_sample(
            w_seg_logits, inv_grid, mode='bilinear', align_corners=False)

        data['seg'] = {'logits': seg_logits, 'label_names': self.label_names}
        return data
    

    def logits2predictions(self, logits: torch.Tensor):
        return logits.argmax(dim=1)

    @torch.no_grad()
    @torch.inference_mode()
    def forward_warped(self, images: torch.Tensor, return_preds: bool = True):
        '''
        images: b x 3 x h x w , torch.uint8, [0, 255]
        '''
        images = self.unify_image_dtype(images)
        seg_logits, _ = self.net(images)  # nfaces x c x h x w
        # seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        if return_preds:
            seg_preds = self.logits2predictions(seg_logits)
            return seg_logits, seg_preds, self.label_names
        else:
            return seg_logits