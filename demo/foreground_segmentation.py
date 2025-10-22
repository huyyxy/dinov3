#!/usr/bin/env python3
"""
使用 DINOv3 特征训练前景分割模型的训练脚本

基于 notebooks/foreground_segmentation.ipynb 实现
"""

import argparse
import io
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm


# 常量定义
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


def load_images_from_tar(tar_path: str) -> list:
    """从 tar.gz 文件中加载图像"""
    images = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if member.isfile():
                image_data = tar.extractfile(member)
                image = Image.open(image_data)
                # 在 tar 文件关闭前强制加载图像数据到内存
                image.load()
                images.append(image)
    return images


def resize_transform(
    mask_image: Image.Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """将图像调整大小到可被补丁大小整除的维度"""
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(
        TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size))
    )


def extract_features_and_labels(
    model,
    images: list,
    labels: list,
    n_layers: int,
    patch_quant_filter: torch.nn.Conv2d,
    device: str = "cuda",
) -> tuple:
    """为所有图像提取特征和标签"""
    xs = []
    ys = []
    image_index = []

    print(f"\n正在处理 {len(images)} 张图像并提取特征...")
    
    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.float32):
            for i in tqdm(range(len(images)), desc="提取特征"):
                # 加载并处理标签
                mask_i = labels[i].split()[-1]
                mask_i_resized = resize_transform(mask_i)
                mask_i_quantized = (
                    patch_quant_filter(mask_i_resized)
                    .squeeze()
                    .view(-1)
                    .detach()
                    .cpu()
                )
                ys.append(mask_i_quantized)

                # 加载并处理图像
                image_i = images[i].convert("RGB")
                image_i_resized = resize_transform(image_i)
                image_i_normalized = TF.normalize(
                    image_i_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD
                )
                image_i_normalized = image_i_normalized.unsqueeze(0).to(device)

                # 提取特征
                feats = model.get_intermediate_layers(
                    image_i_normalized, n=range(n_layers), reshape=True, norm=True
                )
                dim = feats[-1].shape[1]
                xs.append(
                    feats[-1]
                    .squeeze()
                    .view(dim, -1)
                    .permute(1, 0)
                    .detach()
                    .cpu()
                )

                image_index.append(i * torch.ones(ys[-1].shape))

    # 连接所有数据
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    image_index = torch.cat(image_index)

    # 只保留具有明确正标签或负标签的补丁
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    image_index = image_index[idx]

    print(f"特征矩阵大小: {xs.shape}")
    print(f"标签矩阵大小: {ys.shape}")

    return xs, ys, image_index


def cross_validate(
    xs: torch.Tensor,
    ys: torch.Tensor,
    image_index: torch.Tensor,
    n_images: int,
    c_values: np.ndarray,
) -> np.ndarray:
    """使用留一法交叉验证选择最优 C 值"""
    scores = np.zeros((n_images, len(c_values)))

    print("\n开始留一法交叉验证...")
    for i in range(n_images):
        print(f"\n验证图像 {i+1}/{n_images}")
        train_selection = image_index != float(i)
        fold_x = xs[train_selection].numpy()
        fold_y = (ys[train_selection] > 0).long().numpy()
        val_x = xs[~train_selection].numpy()
        val_y = (ys[~train_selection] > 0).long().numpy()

        for j, c in enumerate(c_values):
            print(f"  训练逻辑回归 C={c:.2e}...", end=" ")
            clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(
                fold_x, fold_y
            )
            output = clf.predict_proba(val_x)
            precision, recall, thresholds = precision_recall_curve(
                val_y, output[:, 1]
            )
            s = average_precision_score(val_y, output[:, 1])
            scores[i, j] = s
            print(f"AP={s*100:.1f}")

    return scores


def train_final_model(
    xs: torch.Tensor,
    ys: torch.Tensor,
    c_value: float,
    verbose: bool = True,
) -> LogisticRegression:
    """使用最优 C 值训练最终模型"""
    print(f"\n使用 C={c_value} 训练最终模型...")
    clf = LogisticRegression(
        random_state=0,
        C=c_value,
        max_iter=100000,
        verbose=2 if verbose else 0,
    ).fit(xs.numpy(), (ys > 0).long().numpy())
    print("训练完成！")
    return clf


def main():
    parser = argparse.ArgumentParser(
        description="使用 DINOv3 特征训练前景分割模型"
    )
    parser.add_argument(
        "--images-tar",
        type=str,
        default="datasets/foreground_segmentation_images.tar.gz",
        help="图像 tar.gz 文件路径",
    )
    parser.add_argument(
        "--labels-tar",
        type=str,
        default="datasets/foreground_segmentation_labels.tar.gz",
        help="标签 tar.gz 文件路径",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov3_vitl16",
        choices=list(MODEL_TO_NUM_LAYERS.keys()),
        help="DINOv3 模型名称",
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
        default=".",
        help="输出目录",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="fg_classifier.pkl",
        help="输出模型文件名",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=None,
        help="逻辑回归的 C 值（如果未指定，将通过交叉验证选择）",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="跳过交叉验证，直接使用指定的 C 值",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备类型",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出",
    )

    args = parser.parse_args()

    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        args.device = "cpu"

    # 设置 DINOv3 位置
    if args.dinov3_location is None:
        dinov3_location = os.getenv("DINOV3_LOCATION", "facebookresearch/dinov3")
    else:
        dinov3_location = args.dinov3_location

    source = "local" if dinov3_location != "facebookresearch/dinov3" else "github"
    print(f"DINOv3 位置: {dinov3_location}")
    print(f"加载源: {source}")

    # 加载模型
    print(f"\n加载 DINOv3 模型: {args.model_name}")
    load_kwargs = {
        "repo_or_dir": dinov3_location,
        "model": args.model_name,
        "source": source,
    }
    
    # 如果指定了权重文件，则使用本地权重
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

    # 加载数据
    print(f"\n从 {args.images_tar} 加载图像...")
    images = load_images_from_tar(args.images_tar)
    print(f"从 {args.labels_tar} 加载标签...")
    labels = load_images_from_tar(args.labels_tar)
    n_images = len(images)
    assert n_images == len(labels), f"图像数量 ({n_images}) 与标签数量 ({len(labels)}) 不匹配"
    print(f"成功加载 {n_images} 张图像和标签")

    # 创建补丁量化滤波器
    patch_quant_filter = torch.nn.Conv2d(
        1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False
    )
    patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

    # 提取特征和标签
    xs, ys, image_index = extract_features_and_labels(
        model, images, labels, n_layers, patch_quant_filter, args.device
    )

    # 确定 C 值
    if args.c_value is None or not args.skip_cv:
        # 交叉验证
        c_values = np.logspace(-7, 0, 8)
        scores = cross_validate(xs, ys, image_index, n_images, c_values)

        # 选择最优 C 值
        mean_scores = scores.mean(axis=0)
        best_c_idx = np.argmax(mean_scores)
        best_c = c_values[best_c_idx]

        print("\n交叉验证结果:")
        for i, c in enumerate(c_values):
            print(f"  C={c:.2e}: 平均 AP={mean_scores[i]*100:.2f}%")
        print(f"\n最优 C 值: {best_c:.2e} (平均 AP={mean_scores[best_c_idx]*100:.2f}%)")

        c_value = best_c
    else:
        c_value = args.c_value
        print(f"\n使用指定的 C 值: {c_value}")

    # 训练最终模型
    clf = train_final_model(xs, ys, c_value, args.verbose)

    # 保存模型
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / args.output_name

    print(f"\n保存模型到 {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print("模型保存成功！")

    print("\n训练完成！")
    print(f"模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
