# EDSR-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921).

### Table of contents

- [EDSR-PyTorch](#edsr-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Enhanced Deep Residual Networks for Single Image Super-Resolution](#about-enhanced-deep-residual-networks-for-single-image-super-resolution)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
        - [Download train dataset](#download-train-dataset)
        - [Download valid dataset](#download-valid-dataset)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Enhanced Deep Residual Networks for Single Image Super-Resolution](#enhanced-deep-residual-networks-for-single-image-super-resolution)

## About Enhanced Deep Residual Networks for Single Image Super-Resolution

If you're new to EDSR, here's an abstract straight from the paper:

Recent research on super-resolution has progressed with the development of deep convolutional neural networks
(DCNN). In particular, residual learning techniques exhibit improved performance. In this paper, we develop an enhanced deep super-resolution
network (EDSR) with performance exceeding those of current state-of-the-art SR methods. The significant performance improvement of our model is due to
optimization by removing unnecessary modules in conventional residual networks. The performance is further improved by expanding the model size while
we stabilize the training procedure. We also propose a new multi-scale deep super-resolution system (MDSR) and training method, which can reconstruct
high-resolution images of different upscaling factors in a single model. The proposed methods show superior performance over the state-of-the-art
methods on benchmark datasets and prove its excellence by winning the NTIRE2017 Super-Resolution Challenge.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

### Download train dataset

#### DIV2K

- Image format
    - [Google Driver](https://drive.google.com/drive/folders/12kdXxN2Y2EtUmf0Ob4zhOJBZwg_eUj3J?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1EXXbhxxRDtqPosT2WL8NkA) access: `llot`

### Download valid dataset

#### Set5

- Image format
    - [Google Driver](https://drive.google.com/file/d/1GtQuoEN78q3AIP8vkh-17X90thYp_FfU/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1dlPcpwRPUBOnxlfW5--S5g) access:`llot`

#### Set14

- Image format
    - [Google Driver](https://drive.google.com/file/d/1CzwwAtLSW9sog3acXj8s7Hg3S7kr2HiZ/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1KBS38UAjM7bJ_e6a54eHaA) access:`llot`

#### BSD200

- Image format
    - [Google Driver](https://drive.google.com/file/d/1cdMYTPr77RdOgyAvJPMQqaJHWrD5ma5n/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1xahPw4dNNc3XspMMOuw1Bw) access:`llot`

## Test

Modify the contents of the file as follows.

- line 25: `upscale_factor` change to the magnification you need to enlarge.
- line 27: `mode` change Set to valid mode.
- line 66: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 25: `upscale_factor` change to the magnification you need to enlarge.
- line 27: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 42: `resume` change to `True`.
- line 43: `strict` Transfer learning is set to `False`, incremental learning is set to `True`.
- line 44: `start_epoch` change number of training iterations in the previous round.
- line 45: `resume_weight` the weight address that needs to be loaded.

## Result

Source of original paper results: https://arxiv.org/pdf/1707.02921.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       |
|:-------:|:-----:|:----------------:|
|  DIV2K  |   2   | 34.61(**34.35**) |
|  DIV2K  |   3   | 30.92(**30.84**) |
|  DIV2K  |   4   | 28.95(**28.85**) |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

### Credit

#### Enhanced Deep Residual Networks for Single Image Super-Resolution

_Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu_ <br>

**Abstract** <br>
Recent research on super-resolution has progressed with the development of deep convolutional neural networks
(DCNN). In particular, residual learning techniques exhibit improved performance. In this paper, we develop an enhanced deep super-resolution
network (EDSR) with performance exceeding those of current state-of-the-art SR methods. The significant performance improvement of our model is due to
optimization by removing unnecessary modules in conventional residual networks. The performance is further improved by expanding the model size while
we stabilize the training procedure. We also propose a new multi-scale deep super-resolution system (MDSR) and training method, which can reconstruct
high-resolution images of different upscaling factors in a single model. The proposed methods show superior performance over the state-of-the-art
methods on benchmark datasets and prove its excellence by winning the NTIRE2017 Super-Resolution Challenge.

[[Paper]](https://arxiv.org/pdf/1707.02921)

```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```
