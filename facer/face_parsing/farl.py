from typing import Optional

import torch
import torch.nn as nn

from .base import FaceParser

model_entries = {
    'farl/celebm': 'jit_models/farl/celebm/main_ema_181500_jit.pt'
}

class FaRLFaceParser(FaceParser):
    def __init__(self, conf_name: Optional[str] = None,
                 model_path: Optional[str] = None) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = 'farl/celebm'
        self.net = torch.jit.load(model_entries[conf_name])

    def forward(self, images, data):
        pass