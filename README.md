# FACER

Face related toolkit. This repo is still under construction to include more models.

## Updates

- [27/04/2023] Face parsing model trained on CelebM dataset is available, check it out [here](https://github.com/FacePerceiver/facer/blob/main/samples/face_parsing.ipynb).

## Install

The easiest way to install it is using pip:

```bash
pip install git+https://github.com/FacePerceiver/facer.git@main
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
@inproceedings{zheng2022farl,
  title={General facial representation learning in a visual-linguistic manner},
  author={Zheng, Yinglin and Yang, Hao and Zhang, Ting and Bao, Jianmin and Chen, Dongdong and Huang, Yangyu and Yuan, Lu and Chen, Dong and Zeng, Ming and Wen, Fang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18697--18709},
  year={2022}
}
``` 

