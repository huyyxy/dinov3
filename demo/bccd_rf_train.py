#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
血细胞分类训练脚本 - 使用DINOv3特征 + 随机森林分类器
这种方法更简单直接，通常在小数据集上效果更好
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import time

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# 类别映射
CLASS_NAMES = ['Platelets', 'RBC', 'WBC']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_xml_annotation(xml_file: Path) -> List[Dict]:
    """解析PASCAL VOC格式的XML标注文件
    
    Args:
        xml_file: XML文件路径
        
    Returns:
        标注对象列表，每个对象包含class_name和bbox
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'class_name': class_name,
            'bbox': (xmin, ymin, xmax, ymax)
        })
    
    return objects


def load_dinov3_model(model_name: str = 'dinov3_vits16', weights_path: str = None, 
                      device='cuda', local_repo: str = None):
    """加载DINOv3模型用于特征提取
    
    Args:
        model_name: 模型名称
        weights_path: 预训练权重路径
        device: 设备
        local_repo: 本地仓库路径，如果提供则从本地加载
        
    Returns:
        DINOv3模型
    """
    print(f"📦 加载DINOv3模型: {model_name}")
    
    # 如果提供了本地仓库路径，从本地加载
    if local_repo and os.path.exists(local_repo):
        print(f"📂 使用本地仓库: {local_repo}")
        model = torch.hub.load(
            repo_or_dir=local_repo,
            model=model_name,
            source='local',
            pretrained=False  # 稍后手动加载权重
        )
    else:
        # 从GitHub加载
        model = torch.hub.load(
            repo_or_dir='facebookresearch/dinov3',
            model=model_name,
            source='github',
            pretrained=True  # 使用预训练权重！
        )
    
    # 加载预训练权重
    if weights_path and os.path.exists(weights_path):
        print(f"📦 加载预训练权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        model_keys = set(model.state_dict().keys())
        filtered_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('backbone.', '').replace('module.', '')
            if new_k in model_keys:
                filtered_dict[new_k] = v
        
        model.load_state_dict(filtered_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    print("✓ DINOv3模型加载完成")
    
    return model


def extract_patch_features(model, image: Image.Image, bbox: Tuple[int, int, int, int], 
                          patch_size: int = 224, device='cuda') -> np.ndarray:
    """从图像中提取单个bbox的特征
    
    Args:
        model: DINOv3模型
        image: PIL图像
        bbox: 边界框 (xmin, ymin, xmax, ymax)
        patch_size: 输入patch大小
        device: 设备
        
    Returns:
        特征向量
    """
    xmin, ymin, xmax, ymax = bbox
    
    # 裁剪并resize bbox区域
    patch = image.crop((xmin, ymin, xmax, ymax))
    
    # 转换为tensor
    transform = T.Compose([
        T.Resize((patch_size, patch_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    patch_tensor = transform(patch).unsqueeze(0).to(device)
    
    # 提取特征
    with torch.no_grad():
        features = model.forward_features(patch_tensor)
        
        # 使用CLS token的特征
        if hasattr(features, 'x_norm_clstoken'):
            feat = features.x_norm_clstoken
        elif hasattr(features, 'tokens'):
            feat = features.tokens[:, 0]  # CLS token
        else:
            # 如果是字典形式的输出
            if isinstance(features, dict):
                if 'x_norm_clstoken' in features:
                    feat = features['x_norm_clstoken']
                else:
                    # 尝试使用完整的patch tokens的平均值
                    feat = features.get('x_norm_patchtokens', features.get('tokens', None))
                    if feat is not None and feat.dim() > 2:
                        feat = feat.mean(dim=1)
            else:
                # 假设是单个tensor，取第一个token
                if features.dim() == 3:
                    feat = features[:, 0]
                else:
                    feat = features
        
        # 转换为numpy
        feat = feat.cpu().numpy().flatten()
    
    return feat


def extract_dataset_features(model, data_dir: Path, annotations_dir: Path, 
                             split: str = 'train', device='cuda', 
                             max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """从数据集中提取所有bbox的特征
    
    Args:
        model: DINOv3模型
        data_dir: 图像目录
        annotations_dir: 标注目录
        split: 数据集划分 ('train' 或 'test')
        device: 设备
        max_samples: 最大样本数（用于快速测试）
        
    Returns:
        (features, labels) - 特征矩阵和标签数组
    """
    print(f"\n📊 提取{split}集特征...")
    
    # 获取所有图像文件
    image_files = sorted(list(data_dir.glob('*.jpg')))
    if max_samples:
        image_files = image_files[:max_samples]
    
    features_list = []
    labels_list = []
    
    for img_file in tqdm(image_files, desc=f"处理{split}集"):
        # 查找对应的XML标注文件
        xml_file = annotations_dir / f"{img_file.stem}.xml"
        if not xml_file.exists():
            continue
        
        # 解析标注
        annotations = parse_xml_annotation(xml_file)
        if not annotations:
            continue
        
        # 加载图像
        image = Image.open(img_file).convert('RGB')
        
        # 提取每个bbox的特征
        for obj in annotations:
            class_name = obj['class_name']
            if class_name not in CLASS_TO_IDX:
                continue
            
            bbox = obj['bbox']
            
            try:
                # 提取特征
                feat = extract_patch_features(model, image, bbox, device=device)
                features_list.append(feat)
                labels_list.append(CLASS_TO_IDX[class_name])
            except Exception as e:
                print(f"警告: 提取特征失败 {img_file.name} - {class_name}: {e}")
                continue
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"✓ 提取完成: {len(features)} 个样本")
    print(f"  特征维度: {features.shape}")
    print(f"  类别分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return features, labels


def balance_dataset(X: np.ndarray, y: np.ndarray, 
                   method: str = 'undersample', 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """平衡数据集
    
    Args:
        X: 特征
        y: 标签
        method: 平衡方法 ('undersample' 或 'oversample')
        random_state: 随机种子
        
    Returns:
        平衡后的特征和标签
    """
    from collections import Counter
    
    # 验证输入数据
    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"❌ 输入数据为空！X shape: {X.shape}, y shape: {y.shape}")
    
    if len(X) != len(y):
        raise ValueError(f"❌ X 和 y 的长度不匹配！X: {len(X)}, y: {len(y)}")
    
    np.random.seed(random_state)
    
    # 统计每个类别的样本数
    unique, counts = np.unique(y, return_counts=True)
    
    if len(unique) == 0:
        raise ValueError(f"❌ 没有找到任何类别！请检查标签数据")
    
    class_counts = dict(zip(unique, counts))
    
    print(f"\n⚖️  平衡数据集 (方法: {method})")
    print(f"  原始分布: {class_counts}")
    
    if method == 'undersample':
        # 下采样到最小类别的数量
        min_samples = min(counts)
        
        balanced_X = []
        balanced_y = []
        
        for class_idx in unique:
            class_mask = (y == class_idx)
            class_X = X[class_mask]
            class_y = y[class_mask]
            
            # 随机采样
            indices = np.random.choice(len(class_X), min_samples, replace=False)
            balanced_X.append(class_X[indices])
            balanced_y.append(class_y[indices])
        
        X_balanced = np.vstack(balanced_X)
        y_balanced = np.hstack(balanced_y)
        
    elif method == 'oversample':
        # 上采样到最大类别的数量
        max_samples = max(counts)
        
        balanced_X = []
        balanced_y = []
        
        for class_idx in unique:
            class_mask = (y == class_idx)
            class_X = X[class_mask]
            class_y = y[class_mask]
            
            # 随机采样（允许重复）
            indices = np.random.choice(len(class_X), max_samples, replace=True)
            balanced_X.append(class_X[indices])
            balanced_y.append(class_y[indices])
        
        X_balanced = np.vstack(balanced_X)
        y_balanced = np.hstack(balanced_y)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 打乱数据
    shuffle_indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]
    
    balanced_counts = dict(zip(*np.unique(y_balanced, return_counts=True)))
    print(f"  平衡后分布: {balanced_counts}")
    print(f"  总样本: {len(X_balanced)} (原始: {len(X)})")
    
    return X_balanced, y_balanced


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       n_estimators: int = 200, max_depth: int = 30,
                       random_state: int = 42,
                       balance_method: str = None) -> RandomForestClassifier:
    """训练随机森林分类器
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征（可选）
        y_val: 验证标签（可选）
        n_estimators: 树的数量
        max_depth: 最大深度
        random_state: 随机种子
        balance_method: 数据平衡方法 ('undersample', 'oversample', None)
        
    Returns:
        训练好的随机森林分类器
    """
    # 平衡数据集
    if balance_method:
        X_train, y_train = balance_dataset(X_train, y_train, 
                                          method=balance_method, 
                                          random_state=random_state)
    
    print(f"\n🌲 训练随机森林分类器...")
    print(f"  参数: n_estimators={n_estimators}, max_depth={max_depth}")
    
    start_time = time.time()
    
    # 创建并训练随机森林
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1,
        class_weight='balanced',  # 处理类别不平衡
        min_samples_split=5,  # 减少过拟合
        min_samples_leaf=2
    )
    
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"✓ 训练完成 (用时: {train_time:.2f}秒)")
    
    # 评估训练集
    train_acc = clf.score(X_train, y_train)
    print(f"\n训练集准确率: {train_acc:.4f}")
    
    y_train_pred = clf.predict(X_train)
    print("\n训练集分类报告:")
    print(classification_report(y_train, y_train_pred, target_names=CLASS_NAMES))
    
    # 如果提供了验证集，评估验证集
    if X_val is not None and y_val is not None:
        val_acc = clf.score(X_val, y_val)
        print(f"\n验证集准确率: {val_acc:.4f}")
        
        y_val_pred = clf.predict(X_val)
        print("\n验证集分类报告:")
        print(classification_report(y_val, y_val_pred, target_names=CLASS_NAMES))
        
        print("\n验证集混淆矩阵:")
        print(confusion_matrix(y_val, y_val_pred))
    
    # 特征重要性
    print("\n特征重要性统计:")
    print(f"  平均重要性: {clf.feature_importances_.mean():.6f}")
    print(f"  最大重要性: {clf.feature_importances_.max():.6f}")
    print(f"  最小重要性: {clf.feature_importances_.min():.6f}")
    
    return clf


def main():
    parser = argparse.ArgumentParser(description='血细胞分类训练 - DINOv3 + 随机森林')
    
    # 数据路径
    parser.add_argument('--data_root', type=str, default='../BCCD_Dataset',
                       help='BCCD数据集根目录')
    parser.add_argument('--output_dir', type=str, default='output/bccd_rf_model',
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--dinov3_model', type=str, default='dinov3_vits16',
                       choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vitg14',
                               'dinov3_vith16plus', 'dinov3_vit7b16', 'dinov3_vits16plus', 'dinov3_vitl16plus'],
                       help='DINOv3模型类型')
    parser.add_argument('--dinov3_weights', type=str, default=None,
                       help='DINOv3预训练权重路径')
    parser.add_argument('--patch_size', type=int, default=224,
                       help='输入patch大小')
    
    # 随机森林参数
    parser.add_argument('--n_estimators', type=int, default=200,
                       help='随机森林树的数量')
    parser.add_argument('--max_depth', type=int, default=30,
                       help='随机森林最大深度')
    parser.add_argument('--balance_method', type=str, default='oversample',
                       choices=['undersample', 'oversample', 'none'],
                       help='数据平衡方法：undersample(下采样), oversample(上采样), none(不平衡)')
    
    # 训练参数
    parser.add_argument('--local_repo', type=str, default='/data/william/Workspace/dinov3',
                       help='本地DINOv3仓库路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（用于快速测试）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print("="*80)
    print("血细胞分类训练 - DINOv3特征 + 随机森林分类器")
    print("="*80)
    print(f"数据集: {args.data_root}")
    print(f"DINOv3模型: {args.dinov3_model}")
    print(f"随机森林参数: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"数据平衡方法: {args.balance_method}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    print("="*80)
    
    # 数据路径
    data_root = Path(args.data_root)
    train_img_dir = data_root / 'BCCD' / 'JPEGImages'
    train_ann_dir = data_root / 'BCCD' / 'Annotations'
    test_img_dir = data_root / 'BCCD_Dataset' / 'BCCD' / 'JPEGImages'
    test_ann_dir = data_root / 'BCCD_Dataset' / 'BCCD' / 'Annotations'
    
    # 如果测试集路径不存在，使用训练集路径
    if not test_img_dir.exists():
        test_img_dir = train_img_dir
        test_ann_dir = train_ann_dir
        print("注意: 使用训练集作为验证集")
    
    # 加载DINOv3模型
    dinov3_model = load_dinov3_model(
        model_name=args.dinov3_model,
        weights_path=args.dinov3_weights,
        device=args.device,
        local_repo=args.local_repo
    )
    
    # 提取训练集特征
    X_train, y_train = extract_dataset_features(
        dinov3_model, 
        train_img_dir, 
        train_ann_dir,
        split='train',
        device=args.device,
        max_samples=args.max_samples
    )
    
    # 验证训练数据
    if X_train is None or y_train is None or len(X_train) == 0 or len(y_train) == 0:
        print(f"\n❌ 错误：没有提取到训练数据！")
        print(f"  请检查：")
        print(f"  1. 训练图片目录是否存在且包含图片: {train_img_dir}")
        print(f"  2. 标注目录是否存在且包含标注文件: {train_ann_dir}")
        print(f"  3. 图片和标注文件的格式是否正确")
        return
    
    print(f"\n✅ 成功提取训练特征: {len(X_train)} 个样本")
    
    # 提取验证集特征
    X_val, y_val = None, None
    if test_img_dir != train_img_dir:
        X_val, y_val = extract_dataset_features(
            dinov3_model,
            test_img_dir,
            test_ann_dir,
            split='test',
            device=args.device,
            max_samples=args.max_samples
        )
        if X_val is not None and len(X_val) > 0:
            print(f"✅ 成功提取验证特征: {len(X_val)} 个样本")
    
    # 应用数据平衡（在训练之前）
    balance_method = None if args.balance_method == 'none' else args.balance_method
    if balance_method:
        print(f"\n⚖️  应用数据平衡方法: {balance_method}")
        X_train, y_train = balance_dataset(
            X_train, y_train,
            method=balance_method,
            random_state=args.random_seed
        )
    
    # 保存特征（平衡后的数据）
    features_path = output_dir / 'features.pkl'
    print(f"\n💾 保存特征到: {features_path}")
    with open(features_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'class_names': CLASS_NAMES,
            'class_to_idx': CLASS_TO_IDX,
            'balance_method': args.balance_method
        }, f)
    
    # 训练随机森林（使用已平衡的数据，不再在函数内部平衡）
    rf_classifier = train_random_forest(
        X_train, y_train,
        X_val, y_val,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_seed,
        balance_method=None  # 数据已经平衡，不需要再平衡
    )
    
    # 保存模型
    model_path = output_dir / 'rf_classifier.pkl'
    print(f"\n💾 保存随机森林模型到: {model_path}")
    joblib.dump(rf_classifier, model_path)
    
    # 保存配置
    config_path = output_dir / 'config.pkl'
    print(f"💾 保存配置到: {config_path}")
    config = {
        'dinov3_model': args.dinov3_model,
        'dinov3_weights': args.dinov3_weights,
        'patch_size': args.patch_size,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'balance_method': args.balance_method,  # 保存平衡方法
        'class_names': CLASS_NAMES,
        'class_to_idx': CLASS_TO_IDX
    }
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    print("\n" + "="*80)
    print("训练完成！")
    print(f"模型保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

