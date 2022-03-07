# FACER

Face related toolkit. This repo is still under construction to include more models.

## Install

The easiest way to install it is using pip:

```bash
pip install pyfacer
```
No extra setup needs, pretrained weights will be downloaded automatically.


## Face Detection

We simply wrap a retinaface detector for easy usage.
Check [this notebook](./samples/face_detect.ipynb).

Please consider citing
```
@inproceedings{deng2020retinaface,
  title={Retinaface: Single-shot multi-level face localisation in the wild},
  author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5203--5212},
  year={2020}
}
```

## Face Parsing

We wrap the [FaRL](https://github.com/faceperceiver/farl) models for face parsing.
Check [this notebook](./samples/face_parsing.ipynb).

Please consider citing
```
@article{zheng2021farl,
  title={General Facial Representation Learning in a Visual-Linguistic Manner},
  author={Zheng, Yinglin and Yang, Hao and Zhang, Ting and Bao, Jianmin and Chen, Dongdong and Huang, Yangyu and Yuan, Lu and Chen, Dong and Zeng, Ming and Wen, Fang},
  journal={arXiv preprint arXiv:2112.03109},
  year={2021}
}
``` 

