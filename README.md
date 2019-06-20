# DensePose: 
**Dense Human Pose Estimation In The Wild**

_Rıza Alp Güler, Natalia Neverova, Iasonas Kokkinos_

Forking from: https://github.com/facebookresearch/DensePose and make slightly modifications.

[[`densepose.org`](https://densepose.org)] [[`arXiv`](https://arxiv.org/abs/1802.00434)] [[`BibTeX`](#CitingDensePose)]

Dense human pose estimation aims at mapping all human pixels of an RGB image to the 3D surface of the human body. 
DensePose-RCNN is implemented in the [Detectron](https://github.com/facebookresearch/Detectron) framework and is powered by [Caffe2](https://github.com/caffe2/caffe2).

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1qfSOkpueo1kVZbXOuQJJhyagKjMgepsz" width="700px" />
</div>


In this repository, we provide the code to train and evaluate DensePose-RCNN. We also provide notebooks to visualize the collected DensePose-COCO dataset and show the correspondences to the SMPL model.

## Installation

Please find installation instructions for Caffe2 and DensePose in [`INSTALL.md`](INSTALL.md), a document based on the [Detectron](https://github.com/facebookresearch/Detectron) installation instructions.

## Inference-Training-Testing

After installation, please see [`GETTING_STARTED.md`](GETTING_STARTED.md)  for examples of inference and training and testing.

## Notebooks

### Visualization of DensePose-COCO annotations:

See [`notebooks/DensePose-COCO-Visualize.ipynb`](notebooks/DensePose-COCO-Visualize.ipynb) to visualize the DensePose-COCO annotations on the images:

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1uYRJkIA24KkJU2i4sMwrKa61P0xtZzHk" width="800px" />
</div>

---

### DensePose-COCO in 3D:

See [`notebooks/DensePose-COCO-on-SMPL.ipynb`](notebooks/DensePose-COCO-on-SMPL.ipynb) to localize the DensePose-COCO annotations on the 3D template ([`SMPL`](http://smpl.is.tue.mpg.de)) model:

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1m32oyMuE7AZd3EOf9k8zHpr75C8bHlYj" width="500px" />
</div>

---
### Visualize DensePose-RCNN Results:

See [`notebooks/DensePose-RCNN-Visualize-Results.ipynb`](notebooks/DensePose-RCNN-Visualize-Results.ipynb) to visualize the inferred DensePose-RCNN Results.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1k4HtoXpbDV9MhuyhaVcxDrXnyP_NX896" width="900px" />
</div>

---
### DensePose-RCNN Texture Transfer:

See [`notebooks/DensePose-RCNN-Texture-Transfer.ipynb`](notebooks/DensePose-RCNN-Texture-Transfer.ipynb) to localize the DensePose-COCO annotations on the 3D template ([`SMPL`](http://smpl.is.tue.mpg.de)) model:

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1r-w1oDkDHYnc1vYMbpXcYBVD1-V3B4Le" width="900px" />
</div>

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

## <a name="CitingDensePose"></a>Citing DensePose

If you use Densepose, please use the following BibTeX entry.

```
  @InProceedings{Guler2018DensePose,
  title={DensePose: Dense Human Pose Estimation In The Wild},
  author={R\{i}za Alp G\"uler, Natalia Neverova, Iasonas Kokkinos},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
  }
```

## 我的说明

source activate Caffe2PY27

注意

1、先编译本项目下的detectron（只要改动了detectron下的代码，都需要重新编译一次，否则对文件/detectron/utils/vis.py等的修改可能无效）：
```
cd $DENSEPOSE && make
```
2、获取DensePose所需的数据文件：
```
cd $DENSEPOSE/DensePoseData
bash get_densepose_uv.sh
```
3、运行脚本：
```
带有关键点检测结果的：
python tools/infer_simple.py \
    --cfg configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ziliInfer/ \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl \
    DensePoseData/demo_data/test_jpg

利用本地权重：
python tools/infer_simple.py \
    --cfg configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ziliInfer/ \
    --image-ext jpg \
    --wts models/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl \
    DensePoseData/demo_data/test_jpg


输出简单结果：
python tools/infer_simple.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ziliInfer_simple/ \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    DensePoseData/demo_data/test_jpg

利用本地权重：
python tools/infer_simple.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ziliInfer_simple/ \
    --image-ext jpg \
    --wts models/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    DensePoseData/demo_data/test_jpg
```
