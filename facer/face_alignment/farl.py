from typing import Optional, Dict, Any
import functools
import torch
import torch.nn.functional as F
from .network import FaRLVisualFeatures, MMSEG_UPerHead, FaceAlignmentTransformer, denormalize_points, heatmap2points
from ..transform import (get_face_align_matrix,
                         make_inverted_tanh_warp_grid, make_tanh_warp_grid)
from .base import FaceAlignment
from ..util import download_jit
import io

pretrain_settings = {
    'ibug300w/448': {
        # inter_ocular 0.028835 epoch 60
        'num_classes': 68,
        'url': "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_alignment.farl.ibug300w.main_ema_jit.pt",
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix,
                                           target_shape=(448, 448), target_face_scale=0.8),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0.0, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0.0, warped_shape=(448, 448)),
        
    },
    'aflw19/448': {
        # diag 0.009329 epoch 15
        'num_classes': 19,
        'url': "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_alignment.farl.aflw19.main_ema_jit.pt",
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix,
                                           target_shape=(448, 448), target_face_scale=0.8),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0.0, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0.0, warped_shape=(448, 448)),
    },
    'wflw/448': {
        # inter_ocular 0.038933 epoch 20
        'num_classes': 98,
        'url': "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_alignment.farl.wflw.main_ema_jit.pt",
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix,
                                           target_shape=(448, 448), target_face_scale=0.8),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0.0, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0.0, warped_shape=(448, 448)),
    },

}


def load_face_alignment_model(model_path: str, num_classes=68):
    backbone = FaRLVisualFeatures("base", None, forced_input_resolution=448, output_indices=None).cpu()
    if "jit" in model_path:
        extra_files = {"backbone": None}
        heatmap_head = download_jit(model_path, map_location="cpu", _extra_files=extra_files)
        backbone_weight_io = io.BytesIO(extra_files["backbone"])
        backbone.load_state_dict(torch.load(backbone_weight_io))
        # print("load from jit")
    else:
        channels = backbone.get_output_channel("base")
        in_channels = [channels] * 4
        num_classes = num_classes
        heatmap_head = MMSEG_UPerHead(in_channels=in_channels, channels=channels, num_classes=num_classes) # this requires mmseg as a dependency
        state = torch.load(model_path,map_location="cpu")["networks"]["main_ema"]
        # print("load from checkpoint")

    main_network = FaceAlignmentTransformer(backbone, heatmap_head, heatmap_act="sigmoid").cpu()

    if "jit" not in model_path:
        main_network.load_state_dict(state, strict=True)

    return main_network



class FaRLFaceAlignment(FaceAlignment):
    """ The face alignment models from [FaRL](https://github.com/FacePerceiver/FaRL).

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

    def __init__(self, conf_name: Optional[str] = None,
                 model_path: Optional[str] = None, device=None) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = 'ibug300w/448'
        if model_path is None:
            model_path = pretrain_settings[conf_name]['url']
        self.conf_name = conf_name

        setting  = pretrain_settings[self.conf_name]
        self.net = load_face_alignment_model(model_path, num_classes = setting["num_classes"])
        if device is not None:
            self.net = self.net.to(device)

        self.heatmap_interpolate_mode = 'bilinear'
        self.eval()

    def forward(self, images: torch.Tensor, data: Dict[str, Any]):
        setting = pretrain_settings[self.conf_name]
        images = images.float() / 255.0 # backbone 自带 normalize
        _, _, h, w = images.shape

        simages = images[data['image_ids']]
        matrix = setting['get_matrix_fn'](data[setting['matrix_src_tag']])
        grid = setting['get_grid_fn'](matrix=matrix, orig_shape=(h, w))
        inv_grid = setting['get_inv_grid_fn'](matrix=matrix, orig_shape=(h, w))

        w_images = F.grid_sample(
            simages, grid, mode='bilinear', align_corners=False)

        _, _, warp_h, warp_w = w_images.shape

        heatmap_acted = self.net(w_images)

        warpped_heatmap = F.interpolate(
                            heatmap_acted, size=(warp_h, warp_w),
                            mode=self.heatmap_interpolate_mode, align_corners=False)

        pred_heatmap = F.grid_sample(
            warpped_heatmap, inv_grid, mode='bilinear', align_corners=False)

        landmark = heatmap2points(pred_heatmap)

        landmark = denormalize_points(landmark, h, w)

        data['alignment'] = landmark

        return data


if __name__=="__main__":
    image = torch.randn(1, 3, 448, 448)

    aligner1 = FaRLFaceAlignment("wflw/448")
    
    x1 =  aligner1.net(image)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jit_path", type=str, default=None)
    args = parser.parse_args()

    if args.jit_path is None:
        exit(0)
        
    net  = aligner1.net.cpu()

    features, _ = net.backbone(image)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net.heatmap_head, example_inputs=[features])

    buffer = io.BytesIO()

    torch.save(net.backbone.state_dict(), buffer)
    
    # Save to file
    torch.jit.save(traced_script_module, args.jit_path,
                   _extra_files={"backbone": buffer.getvalue()})

    aligner2 = FaRLFaceAlignment(model_path=args.jit_path)

    # compare the output
    x2 = aligner2.net(image)
    print(torch.allclose(x1, x2))