#!/usr/bin/env python3
"""
使用训练好的 DINOv3 前景分割模型进行推理

基于 foreground_segmentation.py 训练脚本实现
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


# 常量定义（与训练时保持一致）
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}


def resize_transform(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """将图像调整大小到可被补丁大小整除的维度"""
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(
        TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    )


def extract_features(
    model,
    image: Image.Image,
    n_layers: int,
    device: str = "cuda",
) -> torch.Tensor:
    """为单张图像提取 DINOv3 特征"""
    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.float32):
            # 加载并处理图像
            image = image.convert("RGB")
            image_resized = resize_transform(image)
            image_normalized = TF.normalize(
                image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
            image_normalized = image_normalized.unsqueeze(0).to(device)

            # 提取特征
            feats = model.get_intermediate_layers(
                image_normalized, n=range(n_layers), reshape=True, norm=True
            )
            dim = feats[-1].shape[1]
            
            # 获取特征的空间维度
            _, _, h_patches, w_patches = feats[-1].shape
            
            # 重塑特征为 (num_patches, feature_dim)
            features = (
                feats[-1]
                .squeeze()
                .view(dim, -1)
                .permute(1, 0)
                .detach()
                .cpu()
            )

    return features, h_patches, w_patches


def predict_segmentation(
    classifier,
    features: torch.Tensor,
    h_patches: int,
    w_patches: int,
) -> np.ndarray:
    """使用分类器预测前景分割掩码"""
    # 获取前景概率
    proba = classifier.predict_proba(features.numpy())[:, 1]
    
    # 重塑为图像形状
    mask = proba.reshape(h_patches, w_patches)
    
    return mask


def create_overlay(
    original_image: Image.Image,
    mask: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.5,
) -> Image.Image:
    """创建原始图像和分割掩码的叠加图像"""
    # 调整掩码大小到原始图像大小
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            original_image.size, Image.BILINEAR
        )
    ) / 255.0
    
    # 创建彩色掩码（红色表示前景）
    color_mask = np.zeros((*mask_resized.shape, 3), dtype=np.uint8)
    foreground = mask_resized > threshold
    color_mask[foreground] = [255, 0, 0]  # 红色
    
    # 转换为 PIL 图像
    original_array = np.array(original_image.convert("RGB"))
    color_mask_image = Image.fromarray(color_mask)
    
    # 叠加
    overlay = Image.blend(
        Image.fromarray(original_array),
        color_mask_image,
        alpha=alpha
    )
    
    return overlay, mask_resized


def save_results(
    output_dir: Path,
    filename: str,
    original_image: Image.Image,
    mask: np.ndarray,
    overlay: Image.Image,
    save_mask: bool = True,
    save_overlay: bool = True,
):
    """保存推理结果"""
    base_name = Path(filename).stem
    
    if save_mask:
        # 保存掩码图像
        mask_path = output_dir / f"{base_name}_mask.png"
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image = mask_image.resize(original_image.size, Image.BILINEAR)
        mask_image.save(mask_path)
        print(f"  掩码已保存: {mask_path}")
    
    if save_overlay:
        # 保存叠加图像
        overlay_path = output_dir / f"{base_name}_overlay.png"
        overlay.save(overlay_path)
        print(f"  叠加图已保存: {overlay_path}")


def process_image(
    image_path: str,
    model,
    classifier,
    n_layers: int,
    output_dir: Path,
    threshold: float,
    alpha: float,
    device: str,
    save_mask: bool,
    save_overlay: bool,
):
    """处理单张图像"""
    # 加载图像
    image = Image.open(image_path)
    
    # 提取特征
    features, h_patches, w_patches = extract_features(
        model, image, n_layers, device
    )
    
    # 预测分割掩码
    mask = predict_segmentation(classifier, features, h_patches, w_patches)
    
    # 创建叠加图像
    overlay, mask_resized = create_overlay(image, mask, threshold, alpha)
    
    # 保存结果
    save_results(
        output_dir,
        os.path.basename(image_path),
        image,
        mask_resized,
        overlay,
        save_mask,
        save_overlay,
    )
    
    return mask_resized


def main():
    parser = argparse.ArgumentParser(
        description="使用训练好的 DINOv3 前景分割模型进行推理"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入图像路径或包含图像的目录",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="训练好的分类器 pickle 文件路径",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov3_vitl16",
        choices=list(MODEL_TO_NUM_LAYERS.keys()),
        help="DINOv3 模型名称（必须与训练时使用的模型一致）",
    )
    parser.add_argument(
        "--dinov3-location",
        type=str,
        default=None,
        help="DINOv3 仓库位置（本地路径或 'github'）",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="预训练权重文件路径（可选）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="前景/背景分类阈值（0-1）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="叠加图像的透明度（0-1）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备类型",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        default=True,
        help="保存二值掩码图像",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="保存叠加图像",
    )
    parser.add_argument(
        "--image-extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        help="要处理的图像文件扩展名",
    )

    args = parser.parse_args()

    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        args.device = "cpu"

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 加载分类器
    print(f"\n加载分类器: {args.classifier}")
    with open(args.classifier, "rb") as f:
        classifier = pickle.load(f)
    print("分类器加载完成！")

    # 设置 DINOv3 位置
    if args.dinov3_location is None:
        dinov3_location = os.getenv("DINOV3_LOCATION", "facebookresearch/dinov3")
    else:
        dinov3_location = args.dinov3_location

    source = "local" if dinov3_location != "facebookresearch/dinov3" else "github"
    print(f"DINOv3 位置: {dinov3_location}")
    print(f"加载源: {source}")

    # 加载 DINOv3 模型
    print(f"\n加载 DINOv3 模型: {args.model_name}")
    load_kwargs = {
        "repo_or_dir": dinov3_location,
        "model": args.model_name,
        "source": source,
    }
    
    if args.weights:
        print(f"使用本地权重: {args.weights}")
        load_kwargs["weights"] = args.weights
    
    model = torch.hub.load(**load_kwargs)
    model.to(args.device)
    model.eval()
    print("模型加载完成！")

    # 获取模型层数
    n_layers = MODEL_TO_NUM_LAYERS[args.model_name]
    print(f"模型层数: {n_layers}")

    # 确定要处理的图像文件
    input_path = Path(args.input)
    if input_path.is_file():
        # 单张图像
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        # 目录中的所有图像
        image_paths = []
        for ext in args.image_extensions:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))
        image_paths = [str(p) for p in sorted(set(image_paths))]
    else:
        raise ValueError(f"输入路径不存在: {args.input}")

    if not image_paths:
        raise ValueError(f"在 {args.input} 中未找到图像文件")

    print(f"\n找到 {len(image_paths)} 张图像")
    print("参数设置:")
    print(f"  阈值: {args.threshold}")
    print(f"  透明度: {args.alpha}")
    print(f"  保存掩码: {args.save_mask}")
    print(f"  保存叠加图: {args.save_overlay}")

    # 处理所有图像
    print("\n开始处理图像...")
    for image_path in tqdm(image_paths, desc="推理进度"):
        print(f"\n处理: {os.path.basename(image_path)}")
        try:
            process_image(
                image_path,
                model,
                classifier,
                n_layers,
                output_dir,
                args.threshold,
                args.alpha,
                args.device,
                args.save_mask,
                args.save_overlay,
            )
        except Exception as e:
            print(f"  处理失败: {str(e)}")
            continue

    print("\n推理完成！")
    print(f"结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

