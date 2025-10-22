"""
DINOv3 图像特征提取示例

本脚本演示了如何使用 Hugging Face Transformers 库加载和使用 DINOv3 模型
进行图像特征提取。DINOv3 是一个自监督学习的视觉模型，能够提取高质量的
图像表示，适用于各种计算机视觉任务。

主要功能：
1. 加载预训练的 DINOv3 模型
2. 使用自动图像处理器预处理图像
3. 提取图像的特征表示
4. 输出特征向量的维度信息

作者：William
日期：2024
"""

# 导入必要的库
import torch  # PyTorch 深度学习框架
from transformers import AutoImageProcessor, AutoModel  # Hugging Face 预训练模型加载器
from transformers.image_utils import load_image  # 图像加载工具

# 设置图像路径
# 注意：请确保图像文件存在且路径正确
image_path = "/home/william/tiwater.png"

# 加载图像
# load_image 函数会自动处理不同格式的图像文件（PNG, JPG, JPEG等）
# 并返回 PIL Image 对象
print("正在加载图像...")
image = load_image(image_path)
print(f"图像加载成功，尺寸: {image.size}")

# 设置预训练模型路径
# DINOv3 是一个自监督学习的视觉 Transformer 模型
# 这里使用的是 ViT-H/16+ 架构的预训练模型
pretrained_model_name = "/data/dinov3_repo/dinov3-vith16plus-pretrain-lvd1689m"

# 加载图像处理器
# AutoImageProcessor 会自动选择合适的图像预处理方法
# 包括图像尺寸调整、归一化、数据增强等
print("正在加载图像处理器...")
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

# 加载预训练模型
# AutoModel 会自动加载对应的模型架构
# device_map="auto" 会自动选择最佳的设备（CPU/GPU）来加载模型
print("正在加载 DINOv3 模型...")
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto",  # 自动选择设备
)

# 打印模型信息
print(f"模型已加载到设备: {model.device}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 图像预处理
# processor 会将图像转换为模型所需的张量格式
# return_tensors="pt" 表示返回 PyTorch 张量
print("正在预处理图像...")
inputs = processor(images=image, return_tensors="pt").to(model.device)

# 打印输入张量的形状信息
print(f"输入张量形状: {inputs.pixel_values.shape}")

# 模型推理
# torch.inference_mode() 用于推理模式，禁用梯度计算以提高性能
# 这比 torch.no_grad() 更高效，因为它完全禁用了自动微分
print("正在进行模型推理...")
with torch.inference_mode():
    outputs = model(**inputs)

# 提取池化输出
# DINOv3 模型的输出包含多个部分，pooler_output 是全局特征表示
# 这个特征向量可以用于各种下游任务，如图像分类、检索、相似度计算等
pooled_output = outputs.pooler_output

# 输出结果
print("=" * 50)
print("特征提取完成！")
print(f"池化输出形状: {pooled_output.shape}")
print(f"特征维度: {pooled_output.shape[-1]}")
print(f"特征向量范围: [{pooled_output.min().item():.4f}, {pooled_output.max().item():.4f}]")
print("=" * 50)

# 可选：保存特征向量到文件
# torch.save(pooled_output, "image_features.pt")
# print("特征向量已保存到 image_features.pt")
