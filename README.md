# LV-VIS: Large-Vocabulary Video Instance Segmentation dataset

**[📄[arXiv]](https://arxiv.org/abs/2304.01715)**  &emsp; **[📄[ICCV(Oral)]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Towards_Open-Vocabulary_Video_Instance_Segmentation_ICCV_2023_paper.pdf)** &emsp; **[🔥[Dataset Download]](https://github.com/haochenheheda/LVVIS/tree/main?tab=readme-ov-file#dataset-structure)** &emsp; **[🔥[Evaluation Server]](https://www.codabench.org/competitions/1748/)**



This repo is the official implementation of `Towards Open Vocabulary Video Instance Segmentation (ICCV2023 oral)`

## News
We are working on the final revision of the annotations. The Codalab set will be released in November.

## [Towards Open Vocabulary Video Instance Segmentation](https://arxiv.org/pdf/2304.01715.pdf)

>[Haochen Wang](https://scholar.google.com/citations?user=WTZX3y8AAAAJ&hl)<sup>1</sup>\, Cilin Yan<sup>2</sup>\, Shuai Wang <sup>1</sup>\, Xiaolong Jiang <sup>3</sup>\, Xu Tang<sup>3</sup>\, Yao Hu<sup>3</sup>, Weidi Xie <sup>4</sup>,Efstratios Gavves <sup>1</sup>

><sup>1</sup>University of Amsterdam, <sup>2</sup>Beihang University, <sup>3</sup>Xiaohongshu Inc, <sup>4</sup> Shanghai Jiao Tong University.

## LV-VIS dataset

LV-VIS is a dataset/benchmark for Open-Vocabulary Video Instance Segmentation. It contains a total of 4,828 videos with pixel-level segmentation masks for 26,099 objects from 1,196 unique categories.

<img src="assert/dataset/visualizations/00000.gif" alt="Demo" width="240" height="140"> <img src="assert/dataset/visualizations/00001.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00005.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00012.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00013.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00018.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00028.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00035.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00058.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00066.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00078.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00087.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00119.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00129.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00199.gif" alt="Demo" width="240" height="140">
<img src="assert/dataset/visualizations/00203.gif" alt="Demo" width="240" height="140">

### Dataset Download

<!-- 
- [Training Videos](https://drive.google.com/file/d/1er2lBQLF75TI5O4wzGyur0YYoohMK6C3/view?usp=sharing)
- [Validation Videos](https://drive.google.com/file/d/1vTYUz_XLOBnYb9e7upJsZM-nQz2S6wDn/view?usp=drive_link)
- [Test Videos](https://drive.google.com/file/d/13Hgz2hxOPbe4_yTiUpwWb2ZWphaP06AF/view?usp=drive_link)
- [Training Annotations](https://drive.google.com/file/d/18ifd40HuXbjKBtwpUzmboucmOSuAzD1n/view?usp=sharing)
- [Validation Annotations](https://drive.google.com/file/d/1hvZHShzVNmxIQrGGB1chZTV2nqGShi6X/view?usp=drive_link)
- [Validation Annotations (oracle)](https://drive.google.com/file/d/1jmD4aQoP98nOdo_mjtV1eiCAsFGORN0R/view?usp=sharing)
# iccv23
| | Videos | Annotations | Annotations (oracle) |
| :---: | :---: | :---: | :---: |
| Training | [Download](https://drive.google.com/file/d/1er2lBQLF75TI5O4wzGyur0YYoohMK6C3/view?usp=sharing) | [Download](https://drive.google.com/file/d/18ifd40HuXbjKBtwpUzmboucmOSuAzD1n/view?usp=sharing) | - |
| Validation | [Download](https://drive.google.com/file/d/1vTYUz_XLOBnYb9e7upJsZM-nQz2S6wDn/view?usp=drive_link) | [Download](https://drive.google.com/file/d/1hvZHShzVNmxIQrGGB1chZTV2nqGShi6X/view?usp=drive_link) | [Download](https://drive.google.com/file/d/1jmD4aQoP98nOdo_mjtV1eiCAsFGORN0R/view?usp=sharing) |
| Test | [Download](https://drive.google.com/file/d/13Hgz2hxOPbe4_yTiUpwWb2ZWphaP06AF/view?usp=drive_link) | - | - |
-->

| | Videos | Annotations |  Annotations (oracle) | Submission Example |
| :---: | :---: | :---: | :---: | :---: |
| Training | [Download](https://drive.google.com/file/d/1er2lBQLF75TI5O4wzGyur0YYoohMK6C3/view?usp=sharing) | [Download](https://drive.google.com/file/d/1k-o8gBMD7m1-fghw-a1iNDZCi2ZZgV9g/view?usp=sharing) |- | - |
| Validation | [Download](https://drive.google.com/file/d/1vTYUz_XLOBnYb9e7upJsZM-nQz2S6wDn/view?usp=drive_link) | [Download](https://drive.google.com/file/d/1stPD818M3gv7zUV3UIZG1Suru7Tk54jo/view?usp=sharing) | [Download](https://drive.google.com/file/d/1LuOptZe2GjXa-9QuYveidS-e8xkqsapV/view?usp=sharing) | -  | 
| Test | [Download](https://drive.google.com/file/d/13Hgz2hxOPbe4_yTiUpwWb2ZWphaP06AF/view?usp=drive_link) | - |  - |[Download](https://drive.google.com/file/d/1SGCDrdV85ptK5zVIgnA7CgklYXfQFKck/view?usp=drive_link) |


### Dataset Structure

```
## JPEGImages

|- train
  |- 00000
    |- 00000.jpg
    |- 00001.jpg
       ...
  |- 00001
    |- 00000.jpg
    |- 00001.jpg
       ...
    ...
|- val
    ...
|- test
    ...

## Annotations
train_instances.json
val_instances.json
image_val_instances.json

```
The annotation files have the same formation as [Youtube-VIS 2019](https://youtube-vos.org/challenge/2019).


## [Annotation Tool](https://github.com/haochenheheda/segment-anything-annotator)
We used this platform for the annotation of LV-VIS.
This platform is a smart video segmentation annotation tool based on [Lableme](https://github.com/wkentaro/labelme), [SAM](https://github.com/facebookresearch/segment-anything), and [STCN](https://github.com/haochenheheda/STCN). See [segment-anything-annotator](https://github.com/haochenheheda/segment-anything-annotator).


## [Baseline](https://github.com/haochenheheda/LVVIS/blob/main/Baseline.md)

We provide our baseline OV2Seg code for LV-VIS.
Please check [Baseline.md](https://github.com/haochenheheda/LVVIS/blob/main/Baseline.md) for more details.

## TODO

- [x] Training and inference code of OV2Seg
- [x] Leaderboard for Val/test set

**NOTE:** 
* We haven't decided to release the annotation file for the test set yet. Please be patient.
* The training set is not exhaustively annotated.
* If you find mistakes in the annotations, please contact us (h.wang3@uva.nl). We will update the annotations.
  
## Cite

```
@inproceedings{wang2023towards,
  title={Towards Open-Vocabulary Video Instance Segmentation},
  author={Wang, Haochen and Yan, Cilin and Wang, Shuai and Jiang, Xiaolong and Tang, XU and Hu, Yao and Xie, Weidi and Gavves, Efstratios},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## Acknowledgement
This repo is built based on [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [Detic](https://github.com/facebookresearch/Detic), thanks for those excellent projects.
