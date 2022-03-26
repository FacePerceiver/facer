from typing import Optional, Tuple
import torch

from .io import read_hwc, write_hwc
from .util import hwc2bchw, bchw2hwc
from .draw import draw_bchw
from .show import show_bchw, show_bhw

from .face_detection import FaceDetector
from .face_parsing import FaceParser


def _split_name(name: str) -> Tuple[str, Optional[str]]:
    if '/' in name:
        detector_type, conf_name = name.split('/', 1)
    else:
        detector_type, conf_name = name, None
    return detector_type, conf_name


def face_detector(name: str, device: torch.device) -> FaceDetector:
    detector_type, conf_name = _split_name(name)
    if detector_type == 'retinaface':
        from .face_detection import RetinaFaceDetector
        return RetinaFaceDetector(conf_name).to(device)
    else:
        raise RuntimeError(f'Unknown detector type: {detector_type}')


def face_parser(name: str, device: torch.device) -> FaceParser:
    parser_type, conf_name = _split_name(name)
    if parser_type == 'farl':
        from .face_parsing import FaRLFaceParser
        return FaRLFaceParser(conf_name, device=device).to(device)
    else:
        raise RuntimeError(f'Unknown parser type: {parser_type}')
