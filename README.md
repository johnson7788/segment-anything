# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)

**分割模型(SAM)**从点或box等输入提示中产生高质量的目标mask，它可以用来为图像中的所有目标产生mask。它已经在一个由1100万张图像和11亿个masking组成的[数据集](https://segment-anything.com/dataset/index.html)上进行了训练，并在各种分割任务上具有强大的zero-shot性能。

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## Installation

该代码需要`python>=3.8`，以及`pytorch>=1.7`和`torchvision>=0.8`。
请按照[这里](https://pytorch.org/get-started/locally/)的说明来安装PyTorch和TorchVision的依赖。
强烈建议同时安装支持CUDA的PyTorch和TorchVision。 
安装Segment Anything：
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```
以下是mask后处理、以COCO格式保存mask、样本笔记本和以ONNX格式导出模型所需的可选依赖项。`jupyter`也是运行样本笔记本所必需的。
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```


## <a name="GettingStarted"></a>Getting Started

首先下载一个[模型checkpoint](#model-checkpoints)。然后，只需几行就可以使用该模型，从给定的提示中获得mask：

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

或为整个图像生成mask：

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```
此外，还可以从命令行中为图像生成mask：

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

See the examples notebooks on [using SAM with prompts](/notebooks/predictor_example.ipynb) and [automatically generating masks](/notebooks/automatic_mask_generator_example.ipynb) for more details.

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## ONNX Export
SAM的轻量级mask解码器可以导出为ONNX格式，这样它就可以在任何支持ONNX运行的环境中运行，
例如在浏览器中运行，如[演示](https://segment-anything.com/demo)中展示的那样。用以下方法导出模型

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```
关于如何通过SAM的主干将图像预处理与使用ONNX模型进行masking预测结合起来，详见[样本笔记本]（https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb）。
建议使用PyTorch的最新稳定版本进行ONNX输出。

## <a name="Models"></a>Model Checkpoints
该模型有三个模型版本，有不同的主干尺寸。这些模型可以通过运行以下命令实例化
```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```
点击下面的链接，下载相应型号的checkpoint。下载后，将其放在任何位置，然后将其路径传递给`checkpoint`参数。

* **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
* `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Dataset
参见[这里](https://ai.facebook.com/datasets/segment-anything/)，了解数据集的概况。
该数据集可以下载[这里](https://ai.facebook.com/datasets/segment-anything-downloads/)。
通过下载数据集，你同意你已经阅读并接受SA-1B数据集研究许可证的条款。

我们将每张图片的masking保存为一个json文件。它可以作为一个字典在Python中加载，格式如下。

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
```
图片ids可以在sa_images_ids.txt中找到，也可以用上述[链接](https://ai.facebook.com/datasets/segment-anything-downloads/)下载。
将COCO RLE格式的mask解码为二进制：
```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```
See [here](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) for more instructions to manipulate masks stored in RLE format.


## License
The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The Segment Anything project was made possible with the help of many contributors (alphabetical):

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## Citing Segment Anything

If you use SAM or SA-1B in your research, please use the following BibTeX entry. 

```
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
