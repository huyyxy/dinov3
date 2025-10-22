#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
血细胞分类推理脚本 - 使用DINOv3特征 + 随机森林分类器
支持两种模式:
1. 单张图像的完整检测（使用滑动窗口）
2. 对已有标注框进行分类
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import xml.etree.ElementTree as ET

import torch
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import joblib
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# 类别颜色
COLORS = {
    'Platelets': (255, 0, 0),      # 红色 - 血小板
    'RBC': (0, 255, 0),            # 绿色 - 红细胞
    'WBC': (0, 0, 255),            # 蓝色 - 白细胞
}


def load_dinov3_model(model_name: str = 'dinov3_vits16', weights_path: str = None, 
                      device='cuda', local_repo: str = None):
    """加载DINOv3模型用于特征提取
    
    Args:
        model_name: 模型名称
        weights_path: 预训练权重路径
        device: 设备
        local_repo: 本地仓库路径，如果提供则从本地加载
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
    """从图像中提取单个bbox的特征"""
    xmin, ymin, xmax, ymax = bbox
    
    # 确保bbox在图像范围内
    img_width, img_height = image.size
    xmin = max(0, min(xmin, img_width))
    ymin = max(0, min(ymin, img_height))
    xmax = max(0, min(xmax, img_width))
    ymax = max(0, min(ymax, img_height))
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
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
            if isinstance(features, dict):
                if 'x_norm_clstoken' in features:
                    feat = features['x_norm_clstoken']
                else:
                    feat = features.get('x_norm_patchtokens', features.get('tokens', None))
                    if feat is not None and feat.dim() > 2:
                        feat = feat.mean(dim=1)
            else:
                if features.dim() == 3:
                    feat = features[:, 0]
                else:
                    feat = features
        
        feat = feat.cpu().numpy().flatten()
    
    return feat


def parse_xml_annotation(xml_file: Path) -> List[Dict]:
    """解析PASCAL VOC格式的XML标注文件"""
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


def generate_sliding_windows(image_size: Tuple[int, int], window_size: int = 64, 
                             stride: int = 32, min_size: int = 20) -> List[Tuple[int, int, int, int]]:
    """生成滑动窗口的边界框
    
    Args:
        image_size: 图像尺寸 (width, height)
        window_size: 窗口大小
        stride: 滑动步长
        min_size: 最小窗口大小
        
    Returns:
        边界框列表 (xmin, ymin, xmax, ymax)
    """
    width, height = image_size
    windows = []
    
    # 多尺度滑动窗口
    scales = [32, 48, 64, 96, 128]
    
    for scale in scales:
        if scale < min_size:
            continue
        
        # 计算步长（窗口大小的一半）
        step = max(scale // 2, 16)
        
        for y in range(0, height - scale + 1, step):
            for x in range(0, width - scale + 1, step):
                windows.append((x, y, x + scale, y + scale))
    
    return windows


def non_max_suppression(boxes: List[Tuple], scores: List[float], 
                       iou_threshold: float = 0.5) -> List[int]:
    """非极大值抑制
    
    Args:
        boxes: 边界框列表
        scores: 置信度分数列表
        iou_threshold: IoU阈值
        
    Returns:
        保留的边界框索引
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def classify_with_annotations(image_path: Path, xml_path: Path, model, rf_classifier, 
                              config: Dict, device='cuda') -> Tuple[Image.Image, List[Dict]]:
    """使用已有标注框进行分类
    
    Args:
        image_path: 图像路径
        xml_path: XML标注文件路径
        model: DINOv3模型
        rf_classifier: 随机森林分类器
        config: 配置字典
        device: 设备
        
    Returns:
        (可视化图像, 检测结果列表)
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 解析标注
    annotations = parse_xml_annotation(xml_path)
    
    # 提取特征并分类
    results = []
    for obj in tqdm(annotations, desc="分类标注框"):
        bbox = obj['bbox']
        
        # 提取特征
        feat = extract_patch_features(model, image, bbox, 
                                     patch_size=config['patch_size'], 
                                     device=device)
        if feat is None:
            continue
        
        # 预测类别
        pred_label = rf_classifier.predict([feat])[0]
        pred_proba = rf_classifier.predict_proba([feat])[0]
        confidence = pred_proba[pred_label]
        
        pred_class = config['class_names'][pred_label]
        true_class = obj['class_name']
        
        results.append({
            'bbox': bbox,
            'predicted_class': pred_class,
            'true_class': true_class,
            'confidence': confidence,
            'correct': pred_class == true_class
        })
    
    # 可视化
    vis_image = visualize_detections(image, results, show_true_labels=True)
    
    return vis_image, results


def detect_sliding_window(image_path: Path, model, rf_classifier, config: Dict,
                         device='cuda', conf_threshold: float = 0.7,
                         nms_threshold: float = 0.3) -> Tuple[Image.Image, List[Dict]]:
    """使用滑动窗口进行检测
    
    Args:
        image_path: 图像路径
        model: DINOv3模型
        rf_classifier: 随机森林分类器
        config: 配置字典
        device: 设备
        conf_threshold: 置信度阈值
        nms_threshold: NMS的IoU阈值
        
    Returns:
        (可视化图像, 检测结果列表)
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size
    
    print(f"图像尺寸: {img_width} x {img_height}")
    
    # 生成滑动窗口
    windows = generate_sliding_windows((img_width, img_height))
    print(f"生成 {len(windows)} 个候选窗口")
    
    # 对每个窗口进行分类
    detections = []
    for bbox in tqdm(windows, desc="处理窗口"):
        # 提取特征
        feat = extract_patch_features(model, image, bbox,
                                     patch_size=config['patch_size'],
                                     device=device)
        if feat is None:
            continue
        
        # 预测
        pred_proba = rf_classifier.predict_proba([feat])[0]
        pred_label = pred_proba.argmax()
        confidence = pred_proba[pred_label]
        
        # 过滤低置信度
        if confidence > conf_threshold:
            detections.append({
                'bbox': bbox,
                'class_idx': pred_label,
                'confidence': confidence
            })
    
    print(f"检测到 {len(detections)} 个高置信度窗口")
    
    # 按类别分别进行NMS
    final_results = []
    for class_idx in range(len(config['class_names'])):
        class_dets = [d for d in detections if d['class_idx'] == class_idx]
        if len(class_dets) == 0:
            continue
        
        boxes = [d['bbox'] for d in class_dets]
        scores = [d['confidence'] for d in class_dets]
        
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=nms_threshold)
        
        for idx in keep_indices:
            final_results.append({
                'bbox': class_dets[idx]['bbox'],
                'predicted_class': config['class_names'][class_idx],
                'confidence': class_dets[idx]['confidence']
            })
    
    print(f"NMS后保留 {len(final_results)} 个检测框")
    
    # 可视化
    vis_image = visualize_detections(image, final_results, show_true_labels=False)
    
    return vis_image, final_results


def visualize_detections(image: Image.Image, results: List[Dict], 
                         show_true_labels: bool = False) -> Image.Image:
    """可视化检测结果
    
    Args:
        image: PIL图像
        results: 检测结果列表
        show_true_labels: 是否显示真实标签
        
    Returns:
        可视化后的图像
    """
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for det in results:
        bbox = det['bbox']
        pred_class = det['predicted_class']
        confidence = det['confidence']
        
        # 获取颜色
        color = COLORS.get(pred_class, (255, 255, 255))
        
        # 画边界框
        draw.rectangle(bbox, outline=color, width=3)
        
        # 准备标签文本
        if show_true_labels and 'true_class' in det:
            true_class = det['true_class']
            correct = det['correct']
            status = "✓" if correct else "✗"
            label = f"{status} {pred_class} (GT: {true_class}) {confidence:.2f}"
        else:
            label = f"{pred_class} {confidence:.2f}"
        
        # 画标签背景
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], bbox[1] - 20), label, fill=(255, 255, 255), font=font)
    
    return vis_image


def main():
    parser = argparse.ArgumentParser(description='血细胞分类推理 - DINOv3 + 随机森林')
    
    # 输入输出
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--xml', type=str, default=None,
                       help='XML标注文件路径（如果提供，则使用标注框进行分类；否则使用滑动窗口）')
    parser.add_argument('--model_dir', type=str, default='output/bccd_rf_model',
                       help='模型目录')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图像路径')
    
    # 推理参数
    parser.add_argument('--conf_threshold', type=float, default=0.7,
                       help='置信度阈值（仅用于滑动窗口模式）')
    parser.add_argument('--nms_threshold', type=float, default=0.3,
                       help='NMS的IoU阈值（仅用于滑动窗口模式）')
    parser.add_argument('--local_repo', type=str, default='/data/william/Workspace/dinov3',
                       help='本地DINOv3仓库路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print("="*80)
    print("血细胞分类推理 - DINOv3特征 + 随机森林分类器")
    print("="*80)
    
    # 加载配置和模型
    model_dir = Path(args.model_dir)
    config_path = model_dir / 'config.pkl'
    rf_path = model_dir / 'rf_classifier.pkl'
    
    print(f"📦 加载配置: {config_path}")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    print(f"📦 加载随机森林模型: {rf_path}")
    rf_classifier = joblib.load(rf_path)
    
    # 加载DINOv3模型
    dinov3_model = load_dinov3_model(
        model_name=config['dinov3_model'],
        weights_path=config.get('dinov3_weights'),
        device=args.device,
        local_repo=args.local_repo
    )
    
    # 确定推理模式
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    if args.xml:
        # 使用标注框进行分类
        xml_path = Path(args.xml)
        if not xml_path.exists():
            print(f"错误: XML文件不存在: {xml_path}")
            return
        
        print(f"\n模式: 使用标注框进行分类")
        print(f"图像: {image_path}")
        print(f"标注: {xml_path}")
        
        vis_image, results = classify_with_annotations(
            image_path, xml_path, dinov3_model, rf_classifier, 
            config, args.device
        )
        
        # 统计结果
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n结果统计:")
        print(f"  总数: {total}")
        print(f"  正确: {correct}")
        print(f"  准确率: {accuracy:.2%}")
        
        # 按类别统计
        from collections import defaultdict
        class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for r in results:
            true_class = r['true_class']
            class_stats[true_class]['total'] += 1
            if r['correct']:
                class_stats[true_class]['correct'] += 1
        
        print(f"\n各类别准确率:")
        for class_name in config['class_names']:
            if class_name in class_stats:
                stats = class_stats[class_name]
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {class_name}: {stats['correct']}/{stats['total']} = {acc:.2%}")
        
    else:
        # 使用滑动窗口进行检测
        print(f"\n模式: 滑动窗口检测")
        print(f"图像: {image_path}")
        print(f"置信度阈值: {args.conf_threshold}")
        print(f"NMS阈值: {args.nms_threshold}")
        
        vis_image, results = detect_sliding_window(
            image_path, dinov3_model, rf_classifier, config,
            args.device, args.conf_threshold, args.nms_threshold
        )
        
        # 统计结果
        from collections import Counter
        class_counts = Counter([r['predicted_class'] for r in results])
        
        print(f"\n检测统计:")
        for class_name in config['class_names']:
            count = class_counts.get(class_name, 0)
            print(f"  {class_name}: {count}")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path('output/bccd_rf_detections')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_detection.png"
    
    vis_image.save(output_path)
    print(f"\n✓ 结果保存到: {output_path}")
    
    print("="*80)
    print("推理完成！")
    print("="*80)


if __name__ == '__main__':
    main()

