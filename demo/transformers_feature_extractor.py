"""
DINOv3 特征提取器演示脚本

本脚本演示如何使用 Hugging Face Transformers 库加载 DINOv3 模型
进行图像特征提取。DINOv3 是一个自监督学习的视觉模型，能够
提取高质量的图像特征表示。

作者: William
用途: 教学演示 - 展示如何使用预训练的 DINOv3 模型进行特征提取
"""

# 导入必要的库
from transformers import pipeline  # Hugging Face 的管道接口，简化模型使用
from transformers.image_utils import load_image  # 图像加载工具，支持多种格式

# 定义图像路径
# 注意：请确保此路径下的图像文件存在
image_path = "/home/william/tiwater.png"

# 加载图像
# load_image 函数会自动处理图像格式转换和预处理
# 支持常见格式：PNG, JPEG, BMP, TIFF 等
image = load_image(image_path)

# 创建特征提取管道
# pipeline 是 Hugging Face 提供的高级接口，简化了模型加载和推理过程
feature_extractor = pipeline(
    model="/data/dinov3_repo/dinov3-vith16plus-pretrain-lvd1689m",  # 本地模型路径
    task="image-feature-extraction",  # 任务类型：图像特征提取
)

# 提取图像特征
# 输入图像，输出特征向量
# DINOv3 模型会输出一个高维特征向量，通常包含 768 或 1024 个维度
features = feature_extractor(image)

# 打印特征向量信息
# 特征向量可以用于：
# 1. 图像相似度计算
# 2. 图像分类
# 3. 图像检索
# 4. 下游任务的输入特征

print("=" * 60)
print("DINOv3 图像特征提取结果")
print("=" * 60)

# 获取特征向量的基本信息
if isinstance(features, list) and len(features) > 0:
    feature_vector = features[0]  # 获取第一个（通常也是唯一的）特征向量
    
    # 处理嵌套列表结构，展平特征向量
    if isinstance(feature_vector, list) and len(feature_vector) > 0 and isinstance(feature_vector[0], list):
        # 如果是嵌套列表，展平它
        flattened_features = []
        for sublist in feature_vector:
            if isinstance(sublist, list):
                flattened_features.extend(sublist)
            else:
                flattened_features.append(sublist)
        feature_vector = flattened_features
    
    print(f"特征向量维度: {len(feature_vector)}")
    print(f"特征向量类型: {type(feature_vector[0])}")
    
    # 确保所有元素都是数值类型
    try:
        numeric_features = [float(x) for x in feature_vector]
        print(f"特征向量范围: [{min(numeric_features):.4f}, {max(numeric_features):.4f}]")
        print(f"特征向量均值: {sum(numeric_features) / len(numeric_features):.4f}")
    except (ValueError, TypeError) as e:
        print(f"特征向量范围: 无法计算 (包含非数值类型)")
        print(f"特征向量均值: 无法计算 (包含非数值类型)")
        print(f"错误信息: {e}")
    
    print()
    
    # 显示前10个特征值作为示例
    print("前10个特征值示例:")
    for i, value in enumerate(feature_vector[:10]):
        try:
            print(f"  特征[{i:2d}]: {float(value):8.4f}")
        except (ValueError, TypeError):
            print(f"  特征[{i:2d}]: {value}")
    
    if len(feature_vector) > 10:
        print(f"  ... (还有 {len(feature_vector) - 10} 个特征值)")
    
    print()
    print("特征向量应用建议:")
    print("• 使用余弦相似度计算图像相似性")
    print("• 作为分类器的输入特征")
    print("• 用于图像检索和匹配")
    print("• 作为其他深度学习模型的输入")
    
else:
    print("特征提取失败或返回格式异常")
    print(f"返回结果类型: {type(features)}")
    print(f"返回结果内容: {features}")

print("=" * 60)