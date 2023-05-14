from typing import Optional, Tuple
import torch

from .io import read_hwc, write_hwc
from .util import hwc2bchw, bchw2hwc, bchw2bhwc, bhwc2bchw, bhwc2hwc
from .draw import draw_bchw, draw_landmarks
from .show import show_bchw, show_bhw

from .face_detection import FaceDetector
from .face_parsing import FaceParser
from .face_alignment import FaceAlignment
from .face_attribute import FaceAttribute


def _split_name(name: str) -> Tuple[str, Optional[str]]:
    if '/' in name:
        detector_type, conf_name = name.split('/', 1)
    else:
        detector_type, conf_name = name, None
    return detector_type, conf_name


def face_detector(name: str, device: torch.device, **kwargs) -> FaceDetector:
    detector_type, conf_name = _split_name(name)
    if detector_type == 'retinaface':
        from .face_detection import RetinaFaceDetector
        return RetinaFaceDetector(conf_name, **kwargs).to(device)
    else:
        raise RuntimeError(f'Unknown detector type: {detector_type}')


def face_parser(name: str, device: torch.device, **kwargs) -> FaceParser:
    parser_type, conf_name = _split_name(name)
    if parser_type == 'farl':
        from .face_parsing import FaRLFaceParser
        return FaRLFaceParser(conf_name, device=device, **kwargs).to(device)
    else:
        raise RuntimeError(f'Unknown parser type: {parser_type}')


def face_aligner(name: str, device: torch.device, **kwargs) -> FaceAlignment:
    aligner_type, conf_name = _split_name(name)
    if aligner_type == 'farl':
        from .face_alignment import FaRLFaceAlignment
        return FaRLFaceAlignment(conf_name, device=device, **kwargs).to(device)
    else:
        raise RuntimeError(f'Unknown aligner type: {aligner_type}')

def face_attr(name: str, device: torch.device, **kwargs) -> FaceAttribute:
    attr_type, conf_name = _split_name(name)
    if attr_type == 'farl':
        from .face_attribute import FaRLFaceAttribute
        return FaRLFaceAttribute(conf_name, device=device, **kwargs).to(device)
    else:
        raise RuntimeError(f'Unknown attribute type: {attr_type}')