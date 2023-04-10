#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright (c) Meta Platforms, Inc. and affiliates.


# # Produces masks from prompts using an ONNX model

# SAM的提示编码器和mask解码器是非常轻量级的，这使得在给定用户输入时可以有效地计算mask。这个笔记展示了一个例子，说明如何以ONNX格式导出和使用模型的这个轻量级组件，使其能够在支持ONNX运行时的各种平台上运行。

# In[4]:


from IPython.display import display, HTML
display(HTML(
"""
<a target="_blank" href="https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
"""
))


# ## Environment Set-up

# If running locally using jupyter, first install `segment_anything` in your environment using the [installation instructions](https://github.com/facebookresearch/segment-anything#installation) in the repository. The latest stable versions of PyTorch and ONNX are recommended for this notebook. If running from Google Colab, set `using_collab=True` below and run the cell. In Colab, be sure to select 'GPU' under 'Edit'->'Notebook Settings'->'Hardware accelerator'.

# In[5]:


using_colab = False


# In[6]:


if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    import sys
    get_ipython().system('{sys.executable} -m pip install opencv-python matplotlib onnx onnxruntime')
    get_ipython().system("{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'")
    
    get_ipython().system('mkdir images')
    get_ipython().system('wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg')
        
    get_ipython().system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')


# ## Set-up

# 注意这个笔记本需要`onnx`和`onnxruntime`这两个可选依赖项，此外还有`opencv-python`和`matplotlib`用于可视化。

# In[ ]:


import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic


# In[ ]:


def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


# ## Export an ONNX model

# 将下面的路径设置为一个SAM模型checkpoint，然后加载模型。这对于导出模型和计算模型的嵌入都是需要的。

# In[ ]:


checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"


# In[ ]:


sam = sam_model_registry[model_type](checkpoint=checkpoint)


# 脚本`segment-anything/scripts/export_onnx_model.py`可用于导出SAM的必要部分。或者，运行下面的代码来导出一个ONNX模型。如果你已经导出了一个模型，请设置下面的路径并跳到下一节。保证导出的ONNX模型与上面设置的checkpoint和模型类型一致。本笔记本希望在导出模型时使用参数 `return_single_mask=True`。

# In[ ]:


onnx_model_path = None  # Set to use an already exported model, then skip to the next section.


# In[ ]:


import warnings

onnx_model_path = "sam_onnx_example.onnx"

onnx_model = SamOnnxModel(sam, return_single_mask=True)

dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
}
output_names = ["masks", "iou_predictions", "low_res_masks"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )    


# 如果需要，该模型还可以被量化和优化。我们发现这样做可以大大改善网络运行时间，而质量性能的变化可以忽略不计。运行下一个单元来量化模型，否则跳到下一节。

# In[ ]:


onnx_model_quantized_path = "sam_onnx_quantized_example.onnx"
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=onnx_model_quantized_path,
    optimize_model=True,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QUInt8,
)
onnx_model_path = onnx_model_quantized_path


# ## Example Image

# In[ ]:


image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()


# ## Using an ONNX model

# 在这里，作为一个例子，我们在CPU上使用python的`onnxruntime`来执行ONNX模型。然而，原则上可以使用任何支持ONNX运行时的平台。启动下面的运行时会话：

# In[ ]:


ort_session = onnxruntime.InferenceSession(onnx_model_path)


# 为了使用ONNX模型，图像必须首先使用SAM图像编码器进行预处理。这是一个较重的过程，最好在GPU上执行。SamPredictor可以正常使用，然后`.get_image_embedding()`将检索出中间特征。

# In[ ]:


sam.to(device='cuda')
predictor = SamPredictor(sam)


# In[ ]:


predictor.set_image(image)


# In[ ]:


image_embedding = predictor.get_image_embedding().cpu().numpy()


# In[ ]:


image_embedding.shape


# ONNX模型的输入签名与`SamPredictor.predict`不同。以下输入必须全部提供。注意点和mask输入的特殊情况。所有输入都是`np.float32`。
# * `image_embeddings`： 来自`predictor.get_image_embedding()'的图像嵌入。有一个长度为1的批索引。 
# * `point_coords`： 稀疏输入提示的坐标，对应于点输入和box输入。box使用两个点进行编码，一个用于左上角，一个用于右下角。*坐标必须已经转换为长边1024.*有一个长度为1的批索引。 
# * `point_labels`： 稀疏输入提示的标签。0是负的输入点，1是正的输入点，2是左上角的box角，3是右下角的box角，-1是一个填充点。* 如果没有box的输入，应该拼接一个标签为-1、坐标为(0.0, 0.0)的填充点。* 
# `mask_input`： 一个形状为1x1x256x256的模型的mask输入。即使没有mask输入，也必须提供这个。在这种情况下，它可以是零。
# * `has_mask_input`： mask输入的一个 表明器。1表示有mask输入，0表示没有mask输入。
# * `orig_im_size`： 输入图像的尺寸，以(H,W)格式，在任何转换之前。  
# 此外，ONNX模型不对输出的mask logits进行阈值处理。要获得二进制masking，请在`sam.mask_threshold`（等于0.0）处设置阈值。

# ### Example point input

# In[ ]:


input_point = np.array([[500, 375]])
input_label = np.array([1])


# Add a batch index, concatenate a padding point, and transform.

# In[ ]:


onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)


# 创建一个空的mask输入和一个unmask的指标。

# In[ ]:


onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)


# Package the inputs to run in the onnx model

# In[ ]:


ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}


# Predict a mask and threshold it.

# In[ ]:


masks, _, low_res_logits = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold


# In[ ]:


masks.shape


# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show() 


# ### Example mask input

# In[ ]:


input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])

# Use the mask output from the previous run. It is already in the correct form for input to the ONNX model.
onnx_mask_input = low_res_logits


# Transform the points as in the previous example.

# In[ ]:


onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)


# The `has_mask_input` indicator is now 1.

# In[ ]:


onnx_has_mask_input = np.ones(1, dtype=np.float32)


# Package inputs, then predict and threshold the mask.

# In[ ]:


ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

masks, _, _ = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold


# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show() 


# ### Example box and point input

# In[ ]:


input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])


# Add a batch index, concatenate a box and point inputs, add the appropriate labels for the box corners, and transform. There is no padding point since the input includes a box input.

# In[ ]:


onnx_box_coords = input_box.reshape(2, 2)
onnx_box_labels = np.array([2,3])

onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)

onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)


# Package inputs, then predict and threshold the mask.

# In[ ]:


onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

masks, _, _ = ort_session.run(None, ort_inputs)
masks = masks > predictor.model.mask_threshold


# In[ ]:


plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

