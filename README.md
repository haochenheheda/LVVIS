# LV-VIS: Large-Vocabulary Video Instance Segmentation dataset

This repo is the official implementation of `Towards Open Vocabulary Video Instance Segmengtation (ICCV2023)`

[`PDF`](https://arxiv.org/pdf/2304.01715.pdf) | [`Project Page`](xx) | [`Leaderboard`](xx)


## LV-VIS dataset

LV-VIS is a dataset/benchmark for Open-Vocabulary Video Instance Segmentation. It contains a total of 4,828 videos with pixel-level segmentation masks for 26,099 objects from 1,196 unique categories.

<img src="visualizations/00000.gif" alt="Demo" width="300" height="180">

<!--
![](.images/gifs/YFCC100M_8.gif) ![](.images/gifs/Charades_5.gif)
-->

### Dataset Download

- [Training Videos](xx)
- [Validation Videos](xx)
- [Test Videos](xx)
- [Training Annotations](xx)
- [Validation Annotations](xx)

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
```
The annotation files have the same formation as [Youtube-VIS 2019](https://youtube-vos.org/challenge/2019).


## [Annotation Tool](https://github.com/haochenheheda/segment-anything-annotator)
We used this platform for the annotation of LV-VIS.
This platform is a smart video segmentation annotation tool based on [Lableme](https://github.com/wkentaro/labelme), [SAM](https://github.com/facebookresearch/segment-anything), and [STCN](https://github.com/haochenheheda/STCN).




## TODO

* Evaluation Code
* Training/Inference code of OV2Seg
* Leaderboard for Val/test set

**NOTE:** 
* We haven't decided to release the annotation file for the test set yet. Please be patient.
* The training set is not exhaustively annotated.
* If you find mistakes in the annotations, please contact us (h.wang3@uva.nl). We will update the annotations.
  
## Cite

```
@article{wang2023towards,
  title={Towards Open-Vocabulary Video Instance Segmentation},
  author={Wang, Haochen and Wang, Shuai and Yan, Cilin and Jiang, Xiaolong and Tang, XU and Hu, Yao and Xie, Weidi and Gavves, Efstratios},
  journal={arXiv preprint arXiv:2304.01715},
  year={2023}
}
```
