#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright (c) Meta Platforms, Inc. and affiliates.


# # Automatically generating object masks with SAM

# 由于SAM可以有效地处理提示信息，因此可以通过在一幅图像上对大量的提示信息进行采样来生成整个图像的mask。这个方法被用来生成数据集SA-1B。  类`SamAutomaticMaskGenerator`实现了这种能力。它的工作方式是在图像上的网格中对单点输入提示进行采样，SAM可以从每个提示中预测多个mask。然后，mask被过滤以保证质量，并使用非最大限度的压制进行重复计算。额外的选项允许进一步提高masking的质量和数量，如在图像的多个作物上运行预测，或对masking进行后处理以去除小的不连接区域和洞。

# In[2]:


from IPython.display import display, HTML
display(HTML(
"""
<a target="_blank" href="https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
"""
))


# ## Environment Set-up

# If running locally using jupyter, first install `segment_anything` in your environment using the [installation instructions](https://github.com/facebookresearch/segment-anything#installation) in the repository. 如果从Google Colab运行，在下面设置`using_collab=True'并运行单元。在Colab中，一定要在'编辑'->'笔记本设置'->'硬件加速器'下选择'GPU'。

# In[3]:


using_colab = False


# In[4]:


if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    import sys
    get_ipython().system('{sys.executable} -m pip install opencv-python matplotlib')
    get_ipython().system("{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'")
    
    get_ipython().system('mkdir images')
    get_ipython().system('wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg')
        
    get_ipython().system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')


# ## Set-up

# In[5]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


# In[6]:


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


# ## Example image

# In[7]:


image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[8]:


plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()


# ## Automatic mask generation

# 要运行自动mask生成，提供一个SAM模型给`SamAutomaticMaskGenerator`类。将下面的路径设置为SAMcheckpoint。建议在CUDA上运行并使用默认模型。

# In[10]:


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)


# To generate masks, just run `generate` on an image.

# In[11]:


masks = mask_generator.generate(image)


# mask生成返回一个超过mask的列表，其中每个mask是一个包含关于mask的各种数据的字典。这些键是： 
# * `segmentation`：mask 
# * `area`：mask的面积（像素） 
# * `bbox`：mask的边界框，XYWH格式 
# * `predicted_iou`：模型自己对mask质量的预测 
# * `point_coords`：生成该mask的采样输入点 
# * `stability_score`：mask质量的额外测量 
# * `crop_box`：用于生成该mask的XYWH格式图像的裁剪

# In[13]:


print(len(masks))
print(masks[0].keys())


# 显示叠加在图像上的所有masking。

# In[28]:


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 


# ## Automatic mask generation options

# 在自动mask生成中，有几个可优化的参数，可以控制点的采样密度以及去除低质量或重复mask的阈值。此外，自动生成可以在图像的裁剪上运行，以便在较小的目标上获得更好的性能，并且后处理可以去除杂散像素和孔洞。下面是一个对更多mask进行采样的配置实例：

# In[24]:


mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)


# In[25]:


masks2 = mask_generator_2.generate(image)


# In[26]:


len(masks2)


# In[27]:


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show() 


# In[ ]:




