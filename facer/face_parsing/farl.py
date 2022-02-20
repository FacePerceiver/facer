from typing import Optional

import torch
import torch.nn as nn

from .base import FaceParser

pretrained_urls = {
    'farl/celebm': 'https://github.com/YANG-H/FaceR-Private/releases/download/v0.0.1/face_parsing.farl.celebm.main_ema_181500_jit.pt'
}




class FaRLFaceParser(FaceParser):
    def __init__(self, conf_name: Optional[str] = None,
                 model_path: Optional[str] = None) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = 'farl/celebm'
        if model_path is None:
            model_path = 
        self.net = torch.jit.load(model_entries[conf_name])

    def forward(self, images, data):
        pass
