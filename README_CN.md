🆕 [2025-09-17] :fire: DINOv3 骨干网络现在受到 [PyTorch Image Models / timm](https://github.com/huggingface/pytorch-image-models/) 库支持，从版本 [1.0.20](https://github.com/huggingface/pytorch-image-models/releases/tag/v1.0.20) 开始

[2025-08-29] DINOv3 骨干网络受到 Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) 库发布版本支持，从版本 [4.56.0](https://github.com/huggingface/transformers/releases/tag/v4.56.0) 开始

[2025-08-14] DINOv3 骨干网络现在在 [Hugging Face Hub](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009) 上可用，并受到 Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) 库[开发版本](https://github.com/huggingface/transformers/)支持

# DINOv3 🦖🦖🦖

**[Meta AI Research, FAIR](https://ai.meta.com/research/)**

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, <br/>
Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, <br/>
Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, <br/>
Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, <br/>
Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, <br/>
Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski

[ :scroll: [`论文`](https://arxiv.org/abs/2508.10104)] [ :newspaper: [`博客`](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)] [ :globe_with_meridians: [`网站`](https://ai.meta.com/dinov3/)] [ :book: [`BibTeX`](#citing-dinov3)]

DINOv3 的参考 PyTorch 实现和模型。详细信息请参见 **[DINOv3](https://arxiv.org/abs/2508.10104)** 论文。

## 概述

<div align="center">
  <img width="1364" height="1024" alt="market" src="https://github.com/user-attachments/assets/1411f491-988e-49cb-95ae-d03fe6e3c268" />

  <i></em><b>高分辨率密集特征。</b><br/>我们可视化使用 DINOv3 输出特征获得的余弦相似度图<br/>，在标记为红色十字的补丁与所有其他补丁之间。</i>
</div>

<br/>

一个多功能视觉基础模型的扩展家族，产生高质量的密集特征，在各种视觉任务上实现出色的性能，包括在广泛的设置中超越专门的先进技术，无需微调

## 预训练模型

:information_source: 请按照下面提供的链接获取所有模型权重的访问权限：一旦被接受，将发送一封电子邮件，其中包含指向所有可用模型权重（骨干网络和适配器）的完整 URL 列表。这些 URL 可以用于：
- 将模型或适配器权重下载到本地文件系统，并通过 `weights` 或 `backbone_weights` 参数将 `torch.hub.load()` 指向这些本地权重，或
- 直接调用 `torch.hub.load()` 通过 `weights` 或 `backbone_weights` 参数从其 URL 下载和加载骨干网络或适配器。

请参见下面的示例代码片段。

:warning: 请使用 `wget` 而不是网络浏览器来下载权重。

在网页数据集（LVD-1689M）上预训练的 ViT 模型：
<table style="margin: auto">
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>预训练<br/>数据集</th>
      <th>下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/16 distilled </td>
      <td align="right">21M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ViT-S+/16 distilled</td>
      <td align="right">29M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ViT-B/16 distilled</td>
      <td align="right">86M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="right">300M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ViT-H+/16 distilled</td>
      <td align="right">840M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ViT-7B/16</td>
      <td align="right">6,716M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
  </tbody>
</table>

在网页数据集（LVD-1689M）上预训练的 ConvNeXt 模型：
<table style="margin: auto">
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>预训练<br/>数据集</th>
      <th>下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ConvNeXt Tiny</td>
      <td align="right">29M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ConvNeXt Small</td>
      <td align="right">50M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ConvNeXt Base</td>
      <td align="right">89M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ConvNeXt Large</td>
      <td align="right">198M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
  </tbody>
</table>

在卫星数据集（SAT-493M）上预训练的 ViT 模型：
<table style="margin: auto">
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>预训练<br/>数据集</th>
      <th>下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="right">300M</td>
      <td align="center">SAT-493M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
    <tr>
      <td>ViT-7B/16</td>
      <td align="right">6,716M</td>
      <td align="center">SAT-493M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
  </tbody>
</table>


### 预训练骨干网络（通过 PyTorch [Hub](https://docs.pytorch.org/docs/stable/hub.html)）

请按照[这里](https://pytorch.org/get-started/locally/)的说明安装 PyTorch（加载模型所需的唯一依赖项）。强烈建议安装支持 CUDA 的 PyTorch。

```python
import torch

REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>

# 在网页图像上预训练的 DINOv3 ViT 模型
dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

# 在网页图像上预训练的 DINOv3 ConvNeXt 模型
dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_small = torch.hub.load(REPO_DIR, 'dinov3_convnext_small', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_base = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_convnext_large = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

# 在卫星图像上预训练的 DINOv3 ViT 模型
dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
```

### 预训练骨干网络（通过 Hugging Face [Transformers](https://huggingface.co/docs/transformers/)）

所有骨干网络都在 Hugging Face Hub 的 [DINOv3](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009) 集合中可用，并通过 Hugging Face [Transformers](https://huggingface.co/docs/transformers/index) 库支持（从版本 4.56.0 开始的发布包）。请参考相应的文档了解用法，但下面是一个简短的示例，演示如何使用 [Pipeline] 或 [AutoModel] 类获取图像嵌入。

```python
from transformers import pipeline
from transformers.image_utils import load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
features = feature_extractor(image)
```

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("池化输出形状:", pooled_output.shape)
```

其中上面的 `model` 和 `pretrained_model_name` 可以是以下之一：
- `facebook/dinov3-vits16-pretrain-lvd1689m`
- `facebook/dinov3-vits16plus-pretrain-lvd1689m`
- `facebook/dinov3-vitb16-pretrain-lvd1689m`
- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- `facebook/dinov3-vit7b16-pretrain-lvd1689m`
- `facebook/dinov3-convnext-base-pretrain-lvd1689m`
- `facebook/dinov3-convnext-large-pretrain-lvd1689m`
- `facebook/dinov3-convnext-small-pretrain-lvd1689m`
- `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
- `facebook/dinov3-vitl16-pretrain-sat493m`
- `facebook/dinov3-vit7b16-pretrain-sat493m`

### 图像变换

对于使用 LVD-1689M 权重的模型（在网页图像上预训练），请使用以下变换（标准 ImageNet 评估变换）：

```python
import torchvision
from torchvision.transforms import v2

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])
```


对于使用 SAT-493M 权重的模型（在卫星图像上预训练），请使用以下变换：


```python
import torchvision
from torchvision.transforms import v2

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])
```

### 预训练头部 - 图像分类

<table style="margin: auto">
  <thead>
    <tr>
      <th>骨干网络</th>
      <th>预训练<br/>数据集</th>
      <th>头部<br/>数据集</th>
      <th>下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-7B/16</td>
      <td align="center">LVD-1689M</td>
      <td align="center">ImageNet</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
  </tbody>
</table>


（完整）分类器模型可以通过 PyTorch Hub 加载：

```python
import torch

# DINOv3
dinov3_vit7b16_lc = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_lc', source="local", weights=<DEPTHER/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)

```

### 预训练头部 - 在 SYNTHMIX 数据集上训练的深度估计器

<table style="margin: auto">
  <thead>
    <tr>
      <th>骨干网络</th>
      <th>预训练<br/>数据集</th>
      <th>头部<br/>数据集</th>
      <th>下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-7B/16</td>
      <td align="center">LVD-1689M</td>
      <td align="center">SYNTHMIX</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
  </tbody>
</table>


```python
depther = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_dd', source="local", weights=<DEPTHER/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)
```

在图像上运行深度估计器的完整示例代码

```python
from PIL import Image
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import colormaps

def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

depther = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_dd', source="local", weights=<DEPTHER/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)

img_size = 1024
img = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        batch_img = batch_img
        depths = depther(batch_img)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(depths[0,0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")

```

#### 复现论文结果

确保 NYU 数据集按照[这个](DATASETS.md#depth-estimation-on-nyu)设置。

启动以下命令以在 NYUv2 上复现我们论文的深度估计结果，使用在 SYNTHMIX 上训练的预训练深度估计器：

```shell
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/depth/run.py \
config=dinov3/eval/depth/configs/config-nyu-synthmix-dpt-inference.yaml \
datasets.root=<PATH/TO/DATASET> \
load_from=dinov3_vit7b16_dd \
--output-dir <PATH/TO/OUTPUT/DIR>
```

注意事项：
- 如果您想在没有 dinov3.run.submit 的情况下启动代码，您可以使用 python 直接或 torchrun：

```shell
PYTHONPATH=. python dinov3/eval/depth/run.py \
config=dinov3/eval/depth/configs/config-nyu-synthmix-dpt-inference.yaml \
datasets.root=<PATH/TO/DATASET> \
load_from=dinov3_vit7b16_dd \
output_dir=<PATH/TO/OUTPUT/DIR>
```

- 也可以使用 `result_config.save_results=true` 保存预测结果。


#### 在 NYUv2 Depth 上的线性深度估计
```shell
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/depth/run.py \
    model.dino_hub=dinov3_vit7b16 \
    config=dinov3/eval/depth/configs/config-nyu.yaml \
    datasets.root=<PATH/TO/DATASET> \
    --output-dir <PATH/TO/OUTPUT/DIR>
```

作业完成后，您将在指定的输出路径目录中找到
- `depth_config.yaml`，包含您训练模型时使用的配置；
- `model_final.pth`，训练结束时的最终线性头部检查点；和
- `results-depth.csv`，包含最终指标。

### 预训练头部 - 在 COCO2017 数据集上训练的检测器

<table style="margin: auto">
  <thead>
    <tr>
      <th>骨干网络</th>
      <th>预训练<br/>数据集</th>
      <th>头部<br/>数据集</th>
      <th>下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-7B/16</td>
      <td align="center">LVD-1689M</td>
      <td align="center">COCO2017</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
  </tbody>
</table>


```python
detector = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_de', source="local", weights=<DETECTOR/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)
```

### 预训练头部 - 在 ADE20K 数据集上训练的分割器

<table style="margin: auto">
  <thead>
    <tr>
      <th>骨干网络</th>
      <th>预训练<br/>数据集</th>
      <th>头部<br/>数据集</th>
      <th>下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-7B/16</td>
      <td align="center">LVD-1689M</td>
      <td align="center">ADE20K</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a></td>
    </tr>
  </tbody>
</table>

```python
segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", weights=<SEGMENTOR/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)
```

使用提供的分割器（ViT-7B + M2F）在 ADE20K 上运行完整推理的示例命令：

```shell
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/segmentation/run.py \
config=dinov3/eval/segmentation/configs/config-ade20k-m2f-inference.yaml  \
datasets.root=<PATH/TO/DATASET> \
load_from=dinov3_vit7b16_ms \
--output-dir <PATH/TO/OUTPUT/DIR>
```

在图像上运行分割器的完整示例代码

```python
import sys
sys.path.append(REPO_DIR)

from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference


def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", weights=<SEGMENTOR/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)

img_size = 896
img  = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        pred_vit7b = segmentor(batch_img)  # 原始预测  
        # 实际分割图
        segmentation_map_vit7b = make_inference(
            batch_img,
            segmentor,
            inference_mode="slide",
            decoder_head_type="m2f",
            rescale_to=(img.size[-1], img.size[-2]),
            n_output_channels=150,
            crop_size=(img_size, img_size),
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(segmentation_map_vit7b[0,0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")
```




### 预训练头部 - 使用 `dino.txt` 的零样本任务

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">骨干网络</th>
      <th>下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="center">
        <a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[链接]</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz">词汇表</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE">词汇表许可证</a>
      </td>
    </tr>
  </tbody>
</table>

（完整）dino.txt 模型可以通过 PyTorch Hub 加载：

```python
import torch
# DINOv3
dinov3_vitl16_dinotxt_tet1280d20h24l, tokenizer = torch.hub.load(REPO_DIR, 'dinov3_vitl16_dinotxt_tet1280d20h24l', weights=<SEGMENTOR/CHECKPOINT/URL/OR/PATH>, backbone_weights=<BACKBONE/CHECKPOINT/URL/OR/PATH>)
```


## 安装

训练和评估代码需要 PyTorch 版本 >= 2.7.1 以及一些其他第三方包。请注意，代码仅在指定版本下进行过测试，并且期望在 Linux 环境中运行。要设置训练和评估所需的所有依赖项，请按照以下说明：

*[micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)* **（推荐）** - 克隆存储库，然后使用提供的环境定义创建并激活 `dinov3` conda 环境：

```shell
micromamba env create -f conda.yaml
micromamba activate dinov3
```

## 开始使用

提供了几个 notebook 来开始应用 DINOv3：
- [补丁特征的 PCA](notebooks/pca.ipynb)：显示 DINOv3 补丁特征在前景对象上的 PCA（论文中的彩虹可视化）[[在 Google Colab 中运行]](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/pca.ipynb)
- [前景分割](notebooks/foreground_segmentation.ipynb)：基于 DINOv3 特征训练线性前景分割模型 [[在 Google Colab 中运行]](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/foreground_segmentation.ipynb)
- [密集和稀疏匹配](notebooks/dense_sparse_matching.ipynb)：基于 DINOv3 特征匹配两个不同图像上对象的补丁 [[在 Google Colab 中运行]](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/dense_sparse_matching.ipynb)
- [分割跟踪](notebooks/segmentation_tracking.ipynb)：使用基于 DINOv3 特征的非参数方法进行视频分割跟踪 [[在 Google Colab 中运行]](https://colab.research.google.com/github/facebookresearch/dinov3/blob/main/notebooks/segmentation_tracking.ipynb)

## 数据准备

### ImageNet-1k

数据集的根目录应包含以下内容：

- `<ROOT>/test/ILSVRC2012_test_00000001.JPEG`
- `<ROOT>/test/[..]`
- `<ROOT>/test/ILSVRC2012_test_00100000.JPEG`
- `<ROOT>/train/n01440764/n01440764_10026.JPEG`
- `<ROOT>/train/[...]`
- `<ROOT>/train/n15075141/n15075141_9993.JPEG`
- `<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG`
- `<ROOT>/val/[...]`
- `<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG`
- `<ROOT>/labels.txt`

提供的数据集实现期望在额外目录下存在一些额外的元数据文件：

- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`

这些元数据文件可以使用以下 Python 代码生成（一次）：

```python
from dinov3.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

请注意，根目录和额外目录不必是不同的目录。

### ImageNet-22k

请调整[数据集类](dinov3/data/datasets/image_net_22k.py)以匹配您的本地设置。

<br />

:warning: 要执行下一节中提供的训练和评估命令，`dinov3` 包应包含在 Python 模块搜索路径中，即简单地在要运行的命令前加上 `PYTHONPATH=.`。

## 训练

### 快速设置：在 ImageNet-1k 上训练 DINOv3 ViT-L/16

在 4 个 H100-80GB 节点（32 个 GPU）的 SLURM 集群环境中使用 submitit 运行 DINOv3 预训练：

```shell
 PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 4 \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=ImageNet22k:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```
训练时间约为 14 小时，生成的检查点应在 k-NN 评估上达到 82.0%，在线性评估上达到 83.5%。

训练代码每 12500 次迭代在评估文件夹中保存教师权重以供评估。

### 精确的 DINOv3 设置：训练 DINOv3 ViT-7B/16

DINOv3 ViT-7B/16 在私有数据集上训练。训练涉及 3 个阶段：
- 预训练
- Gram 锚定
- 高分辨率适应

#### 预训练

在 32 个节点（256 个 GPU）的 SLURM 集群环境中使用 submitit 启动 DINOV3 ViT-7B/16 预训练。

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 32 \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=<DATASET>:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

#### Gram 锚定

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 32 \
  --config-file dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=<DATASET>:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  gram.ckpt=<PATH/TO/GRAM_TEACHER_FROM_PREVIOUS_STEP>   
```

#### 高分辨率适应


```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 32 \
  --config-file dinov3/configs/train/dinov3_vit7b16_high_res_adapt.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  train.dataset_path=<DATASET>:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  gram.ckpt=<PATH/TO/TEACHER_FROM_GRAM> \
  student.resume_from_teacher_chkpt=<PATH/TO/TEACHER_FROM_GRAM>
```

## 多蒸馏 

### 测试设置：

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 1 \
  --config-file dinov3/configs/train/multi_distillation_test.yaml \
  --output-dir <PATH/TO/OUTPUT/DIR> \
  --multi-distillation \
  train.dataset_path=<DATASET>:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

## 评估

训练代码定期保存教师权重。为了评估模型，在单个节点上运行以下评估：


### 在 ImageNet-1k 上的逻辑回归分类

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/log_regression.py \
  model.config_file=<PATH/TO/OUTPUT/DIR>/config.yaml \
  model.pretrained_weights=<PATH/TO/OUTPUT/DIR>/teacher_checkpoint.pth \
  output_dir=<PATH/TO/OUTPUT/DIR> \
  train.dataset=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  eval.test_dataset=ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### 在 ImageNet-1k 上的 k-NN 分类

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/knn.py \
  model.config_file=<PATH/TO/OUTPUT/DIR>/config.yaml \
  model.pretrained_weights=<PATH/TO/OUTPUT/DIR>/teacher_checkpoint.pth \
  output_dir=<PATH/TO/OUTPUT/DIR> \
  train.dataset=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  eval.test_dataset=ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### 在 ImageNet-1k 上使用数据增强的线性分类

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/linear.py \
  model.config_file=<PATH/TO/OUTPUT/DIR>/config.yaml \
  model.pretrained_weights=<PATH/TO/OUTPUT/DIR>/teacher_checkpoint.pth \
  output_dir=<PATH/TO/OUTPUT/DIR> \
  train.dataset=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
  train.val_dataset=ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### 在 ADE20K 上使用数据增强的线性分割

```shell
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/segmentation/run.py \
model.dino_hub=dinov3_vit7b16 \
config=dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
datasets.root=<PATH/TO/DATASET> \
--output-dir <PATH/TO/OUTPUT/DIR>
```

作业完成后，您将在指定的输出路径目录中找到
- `segmentation_config.yaml`，包含您训练模型时使用的配置；
- `model_final.pth`，训练结束时的最终线性头部检查点；和
- `results-semantic-segmentation.csv`，包含最终指标。

### 使用 dino.txt 在 DINOv3 上进行文本对齐

文本对齐可以按照 `dino.txt` 即 [DINOv2 Meets Text](https://arxiv.org/abs/2412.16334) 的方法进行。

```shell
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/text/train_dinotxt.py \
   --nodes 4 \
  # 文本对齐的示例配置在这里：dinov3/eval/text/configs/dinov3_vitl_text.yaml \ 
  trainer_config_file="<PATH/TO/DINOv3/TEXT/CONFIG>" \
  output-dir=<PATH/TO/OUTPUT/DIR>
```
启动上述命令在 4 个节点上训练文本对齐，每个节点 8 个 gpu（总共 32 个 gpu）。
请注意，DINOv3 论文中的文本对齐模型是在私有数据集上训练的，这里我们提供了一个使用 ```CocoCaptions``` 数据集的示例配置 ```dinov3/eval/text/configs/dinov3_vitl_text.yaml``` 用于说明目的。
请调整提供的 ```CocoCaptions``` 数据集类，数据集可以在[这里](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption)找到  

## 许可证

DINOv3 代码和模型权重在 DINOv3 许可证下发布。有关其他详细信息，请参见 [LICENSE.md](LICENSE.md)。

## 贡献

请参见[贡献](CONTRIBUTING.md)和[行为准则](CODE_OF_CONDUCT.md)。

## 引用 DINOv3

如果您发现此存储库有用，请考虑给一个星标 :star: 和引用 :t-rex::

```
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```
