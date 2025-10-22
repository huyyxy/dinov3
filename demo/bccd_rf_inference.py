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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import pickle
import xml.etree.ElementTree as ET

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
from tqdm import tqdm
import joblib

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 全局配置
COLORS = {
    'Platelets': (255, 0, 0),      # 红色 - 血小板
    'RBC': (0, 255, 0),            # 绿色 - 红细胞
    'WBC': (0, 0, 255),            # 蓝色 - 白细胞
}


# ============================================================================
# 1. 数据处理相关函数
# ============================================================================

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


def clip_bbox_to_image(bbox: Tuple[int, int, int, int], 
                       image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """将边界框裁剪到图像范围内
    
    Args:
        bbox: 边界框 (xmin, ymin, xmax, ymax)
        image_size: 图像尺寸 (width, height)
        
    Returns:
        裁剪后的边界框，如果无效则返回None
    """
    xmin, ymin, xmax, ymax = bbox
    img_width, img_height = image_size
    
    xmin = max(0, min(xmin, img_width))
    ymin = max(0, min(ymin, img_height))
    xmax = max(0, min(xmax, img_width))
    ymax = max(0, min(ymax, img_height))
    
    # 检查边界框是否有效
    if xmax <= xmin or ymax <= ymin:
        return None
    
    return (xmin, ymin, xmax, ymax)


def generate_sliding_windows(image_size: Tuple[int, int], 
                            scales: Optional[List[int]] = None,
                            min_size: int = 20) -> List[Tuple[int, int, int, int]]:
    """生成多尺度滑动窗口的边界框
    
    Args:
        image_size: 图像尺寸 (width, height)
        scales: 窗口尺度列表
        min_size: 最小窗口大小
        
    Returns:
        边界框列表 [(xmin, ymin, xmax, ymax), ...]
    """
    width, height = image_size
    
    # 默认尺度
    if scales is None:
        scales = [32, 48, 64, 96, 128]
    
    windows = []
    
    for scale in scales:
        if scale < min_size:
            continue
        
        # 计算步长（窗口大小的一半）
        step = max(scale // 2, 16)
        
        # 生成该尺度的所有窗口
        for y in range(0, height - scale + 1, step):
            for x in range(0, width - scale + 1, step):
                windows.append((x, y, x + scale, y + scale))
    
    return windows


# ============================================================================
# 2. 模型相关函数
# ============================================================================

def load_dinov3_model(model_name: str = 'dinov3_vits16', 
                      weights_path: Optional[str] = None, 
                      device: str = 'cuda', 
                      local_repo: Optional[str] = None) -> torch.nn.Module:
    """加载DINOv3模型用于特征提取
    
    Args:
        model_name: 模型名称
        weights_path: 预训练权重路径
        device: 计算设备
        local_repo: 本地仓库路径，如果提供则从本地加载
        
    Returns:
        DINOv3模型实例
    """
    print(f"📦 加载DINOv3模型: {model_name}")
    
    # 加载模型结构
    model = _load_model_structure(model_name, local_repo)
    
    # 加载预训练权重
    if weights_path and os.path.exists(weights_path):
        _load_pretrained_weights(model, weights_path)
    
    model = model.to(device)
    model.eval()
    print("✓ DINOv3模型加载完成")
    
    return model


def _load_model_structure(model_name: str, local_repo: Optional[str]) -> torch.nn.Module:
    """加载模型结构"""
    if local_repo and os.path.exists(local_repo):
        print(f"📂 使用本地仓库: {local_repo}")
        model = torch.hub.load(
            repo_or_dir=local_repo,
            model=model_name,
            source='local',
            pretrained=False
        )
    else:
        model = torch.hub.load(
            repo_or_dir='facebookresearch/dinov3',
            model=model_name,
            source='github',
            pretrained=True
        )
    
    return model


def _load_pretrained_weights(model: torch.nn.Module, weights_path: str):
    """加载预训练权重到模型"""
    print(f"📦 加载预训练权重: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # 处理不同的权重格式
    if 'teacher' in state_dict:
        state_dict = state_dict['teacher']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    
    # 过滤并加载权重
    model_keys = set(model.state_dict().keys())
    filtered_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('backbone.', '').replace('module.', '')
        if new_k in model_keys:
            filtered_dict[new_k] = v
    
    model.load_state_dict(filtered_dict, strict=False)


# ============================================================================
# 3. 特征提取相关函数
# ============================================================================

def extract_patch_features(model: torch.nn.Module, 
                          image: Image.Image, 
                          bbox: Tuple[int, int, int, int], 
                          patch_size: int = 224, 
                          device: str = 'cuda') -> Optional[np.ndarray]:
    """从图像的单个bbox区域提取DINOv3特征
    
    Args:
        model: DINOv3模型
        image: PIL图像
        bbox: 边界框 (xmin, ymin, xmax, ymax)
        patch_size: 输入patch大小
        device: 计算设备
        
    Returns:
        特征向量（1D numpy array），如果bbox无效则返回None
    """
    # 确保bbox在图像范围内
    clipped_bbox = clip_bbox_to_image(bbox, image.size)
    if clipped_bbox is None:
        return None
    
    # 预处理图像patch
    patch_tensor = _preprocess_image_patch(image, clipped_bbox, patch_size, device)
    
    # 提取特征
    with torch.no_grad():
        features = model.forward_features(patch_tensor)
        feat = _extract_cls_token(features)
        feat = feat.cpu().numpy().flatten()
    
    return feat


def _preprocess_image_patch(image: Image.Image, 
                            bbox: Tuple[int, int, int, int],
                            patch_size: int, 
                            device: str) -> torch.Tensor:
    """预处理图像patch"""
    xmin, ymin, xmax, ymax = bbox
    
    # 裁剪bbox区域
    patch = image.crop((xmin, ymin, xmax, ymax))
    
    # 转换为tensor
    transform = T.Compose([
        T.Resize((patch_size, patch_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    patch_tensor = transform(patch).unsqueeze(0).to(device)
    return patch_tensor


def _extract_cls_token(features) -> torch.Tensor:
    """从DINOv3特征中提取CLS token"""
    # 优先使用x_norm_clstoken
    if hasattr(features, 'x_norm_clstoken'):
        return features.x_norm_clstoken
    
    # 尝试使用tokens的第一个
    if hasattr(features, 'tokens'):
        return features.tokens[:, 0]
    
    # 处理字典格式的输出
    if isinstance(features, dict):
        if 'x_norm_clstoken' in features:
            return features['x_norm_clstoken']
        
        # 使用patch tokens的平均值
        feat = features.get('x_norm_patchtokens', features.get('tokens', None))
        if feat is not None and feat.dim() > 2:
            return feat.mean(dim=1)
    
    # 处理tensor输出
    if features.dim() == 3:
        return features[:, 0]
    
    return features


# ============================================================================
# 4. 检测相关函数
# ============================================================================

def non_max_suppression(boxes: List[Tuple], 
                       scores: List[float], 
                       iou_threshold: float = 0.5) -> List[int]:
    """非极大值抑制（NMS）
    
    Args:
        boxes: 边界框列表 [(xmin, ymin, xmax, ymax), ...]
        scores: 置信度分数列表
        iou_threshold: IoU阈值
        
    Returns:
        保留的边界框索引列表
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 提取坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 计算面积
    areas = (x2 - x1) * (y2 - y1)
    
    # 按分数降序排列
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # 保留当前最高分数的框
        i = order[0]
        keep.append(i)
        
        # 计算IoU
        iou = _compute_iou_vectorized(boxes[i], boxes[order[1:]], areas[i], areas[order[1:]])
        
        # 保留IoU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def _compute_iou_vectorized(box: np.ndarray, 
                            boxes: np.ndarray, 
                            box_area: float, 
                            boxes_area: np.ndarray) -> np.ndarray:
    """向量化计算IoU"""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    
    iou = inter / (box_area + boxes_area - inter)
    return iou


def apply_nms_per_class(detections: List[Dict], 
                       class_names: List[str],
                       iou_threshold: float = 0.3) -> List[Dict]:
    """对每个类别分别应用NMS
    
    Args:
        detections: 检测结果列表
        class_names: 类别名称列表
        iou_threshold: IoU阈值
        
    Returns:
        NMS后的检测结果列表
    """
    final_results = []
    
    # 按类别分组
    for class_idx in range(len(class_names)):
        class_dets = [d for d in detections if d['class_idx'] == class_idx]
        if len(class_dets) == 0:
            continue
        
        boxes = [d['bbox'] for d in class_dets]
        scores = [d['confidence'] for d in class_dets]
        
        # 应用NMS
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=iou_threshold)
        
        # 添加保留的检测结果
        for idx in keep_indices:
            final_results.append({
                'bbox': class_dets[idx]['bbox'],
                'predicted_class': class_names[class_idx],
                'confidence': class_dets[idx]['confidence']
            })
    
    return final_results


# ============================================================================
# 5. 分类和检测流程函数
# ============================================================================

def classify_with_annotations(image_path: Path, 
                             xml_path: Path, 
                             model: torch.nn.Module, 
                             rf_classifier, 
                             config: Dict, 
                             device: str = 'cuda') -> Tuple[Image.Image, List[Dict]]:
    """使用已有标注框进行分类
    
    Args:
        image_path: 图像路径
        xml_path: XML标注文件路径
        model: DINOv3模型
        rf_classifier: 随机森林分类器
        config: 配置字典
        device: 计算设备
        
    Returns:
        (可视化图像, 检测结果列表)
    """
    # 加载图像和标注
    image = Image.open(image_path).convert('RGB')
    annotations = parse_xml_annotation(xml_path)
    
    # 对每个标注框进行分类
    results = []
    for obj in tqdm(annotations, desc="分类标注框"):
        bbox = obj['bbox']
        
        # 提取特征并预测
        prediction = _classify_single_bbox(
            model, rf_classifier, image, bbox, 
            config['patch_size'], config['class_names'], device
        )
        
        if prediction is None:
            continue
        
        # 添加真实标签信息
        results.append({
            'bbox': bbox,
            'predicted_class': prediction['predicted_class'],
            'true_class': obj['class_name'],
            'confidence': prediction['confidence'],
            'correct': prediction['predicted_class'] == obj['class_name']
        })
    
    # 可视化
    vis_image = visualize_detections(image, results, show_true_labels=True)
    
    return vis_image, results


def _classify_single_bbox(model: torch.nn.Module, 
                         rf_classifier, 
                         image: Image.Image, 
                         bbox: Tuple[int, int, int, int],
                         patch_size: int, 
                         class_names: List[str],
                         device: str) -> Optional[Dict]:
    """对单个边界框进行分类"""
    # 提取特征
    feat = extract_patch_features(model, image, bbox, patch_size=patch_size, device=device)
    if feat is None:
        return None
    
    # 预测类别
    pred_label = rf_classifier.predict([feat])[0]
    pred_proba = rf_classifier.predict_proba([feat])[0]
    confidence = pred_proba[pred_label]
    
    return {
        'predicted_class': class_names[pred_label],
        'confidence': confidence
    }


def detect_sliding_window(image_path: Path, 
                         model: torch.nn.Module, 
                         rf_classifier, 
                         config: Dict,
                         device: str = 'cuda', 
                         conf_threshold: float = 0.7,
                         nms_threshold: float = 0.3) -> Tuple[Image.Image, List[Dict]]:
    """使用滑动窗口进行目标检测
    
    Args:
        image_path: 图像路径
        model: DINOv3模型
        rf_classifier: 随机森林分类器
        config: 配置字典
        device: 计算设备
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
    detections = _classify_all_windows(
        model, rf_classifier, image, windows,
        config['patch_size'], device, conf_threshold
    )
    
    print(f"检测到 {len(detections)} 个高置信度窗口")
    
    # 应用NMS
    final_results = apply_nms_per_class(
        detections, config['class_names'], iou_threshold=nms_threshold
    )
    
    print(f"NMS后保留 {len(final_results)} 个检测框")
    
    # 可视化
    vis_image = visualize_detections(image, final_results, show_true_labels=False)
    
    return vis_image, final_results


def _classify_all_windows(model: torch.nn.Module, 
                         rf_classifier, 
                         image: Image.Image, 
                         windows: List[Tuple[int, int, int, int]],
                         patch_size: int, 
                         device: str,
                         conf_threshold: float) -> List[Dict]:
    """对所有窗口进行分类"""
    detections = []
    
    for bbox in tqdm(windows, desc="处理窗口"):
        # 提取特征
        feat = extract_patch_features(model, image, bbox, patch_size=patch_size, device=device)
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
    
    return detections


# ============================================================================
# 6. 可视化相关函数
# ============================================================================

def visualize_detections(image: Image.Image, 
                         results: List[Dict], 
                         show_true_labels: bool = False) -> Image.Image:
    """可视化检测结果
    
    Args:
        image: PIL图像
        results: 检测结果列表
        show_true_labels: 是否显示真实标签（用于评估模式）
        
    Returns:
        可视化后的图像
    """
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    font = _load_font()
    
    for det in results:
        _draw_single_detection(draw, det, font, show_true_labels)
    
    return vis_image


def _load_font() -> ImageFont.FreeTypeFont:
    """加载字体"""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    return font


def _draw_single_detection(draw: ImageDraw.Draw, 
                          detection: Dict, 
                          font: ImageFont.FreeTypeFont,
                          show_true_labels: bool):
    """绘制单个检测结果"""
    bbox = detection['bbox']
    pred_class = detection['predicted_class']
    confidence = detection['confidence']
    
    # 获取颜色
    color = COLORS.get(pred_class, (255, 255, 255))
    
    # 画边界框
    draw.rectangle(bbox, outline=color, width=3)
    
    # 准备标签文本
    label = _format_label(detection, show_true_labels)
    
    # 画标签背景和文本
    text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((bbox[0], bbox[1] - 20), label, fill=(255, 255, 255), font=font)


def _format_label(detection: Dict, show_true_labels: bool) -> str:
    """格式化标签文本"""
    pred_class = detection['predicted_class']
    confidence = detection['confidence']
    
    if show_true_labels and 'true_class' in detection:
        true_class = detection['true_class']
        correct = detection['correct']
        status = "✓" if correct else "✗"
        return f"{status} {pred_class} (GT: {true_class}) {confidence:.2f}"
    else:
        return f"{pred_class} {confidence:.2f}"


# ============================================================================
# 7. 统计和评估相关函数
# ============================================================================

def compute_classification_stats(results: List[Dict]) -> Dict:
    """计算分类统计信息
    
    Args:
        results: 包含true_class和correct字段的结果列表
        
    Returns:
        统计信息字典
    """
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total if total > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy
    }


def compute_per_class_stats(results: List[Dict], class_names: List[str]) -> Dict[str, Dict]:
    """计算每个类别的统计信息
    
    Args:
        results: 包含true_class和correct字段的结果列表
        class_names: 类别名称列表
        
    Returns:
        每个类别的统计信息
    """
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for r in results:
        true_class = r['true_class']
        class_stats[true_class]['total'] += 1
        if r['correct']:
            class_stats[true_class]['correct'] += 1
    
    # 计算每个类别的准确率
    for class_name in class_names:
        if class_name in class_stats:
            stats = class_stats[class_name]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    return dict(class_stats)


def compute_detection_stats(results: List[Dict], class_names: List[str]) -> Dict[str, int]:
    """计算检测统计信息
    
    Args:
        results: 检测结果列表
        class_names: 类别名称列表
        
    Returns:
        每个类别的检测数量
    """
    class_counts = Counter([r['predicted_class'] for r in results])
    
    # 确保所有类别都有计数
    stats = {class_name: class_counts.get(class_name, 0) for class_name in class_names}
    
    return stats


def print_classification_results(stats: Dict, per_class_stats: Dict, class_names: List[str]):
    """打印分类结果统计"""
    print(f"\n结果统计:")
    print(f"  总数: {stats['total']}")
    print(f"  正确: {stats['correct']}")
    print(f"  准确率: {stats['accuracy']:.2%}")
    
    print(f"\n各类别准确率:")
    for class_name in class_names:
        if class_name in per_class_stats:
            class_stat = per_class_stats[class_name]
            print(f"  {class_name}: {class_stat['correct']}/{class_stat['total']} = {class_stat['accuracy']:.2%}")


def print_detection_results(stats: Dict[str, int]):
    """打印检测结果统计"""
    print(f"\n检测统计:")
    for class_name, count in stats.items():
        print(f"  {class_name}: {count}")


# ============================================================================
# 8. 模型加载和配置相关函数
# ============================================================================

def load_model_and_config(model_dir: Path, 
                         device: str,
                         local_repo: Optional[str] = None) -> Tuple[torch.nn.Module, object, Dict]:
    """加载配置、DINOv3模型和随机森林分类器
    
    Args:
        model_dir: 模型目录
        device: 计算设备
        local_repo: 本地仓库路径
        
    Returns:
        (dinov3_model, rf_classifier, config)
    """
    # 加载配置
    config_path = model_dir / 'config.pkl'
    print(f"📦 加载配置: {config_path}")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # 加载随机森林模型
    rf_path = model_dir / 'rf_classifier.pkl'
    print(f"📦 加载随机森林模型: {rf_path}")
    rf_classifier = joblib.load(rf_path)
    
    # 加载DINOv3模型
    dinov3_model = load_dinov3_model(
        model_name=config['dinov3_model'],
        weights_path=config.get('dinov3_weights'),
        device=device,
        local_repo=local_repo
    )
    
    return dinov3_model, rf_classifier, config


def validate_paths(image_path: Path, xml_path: Optional[Path] = None) -> bool:
    """验证输入路径是否存在
    
    Args:
        image_path: 图像路径
        xml_path: XML标注文件路径（可选）
        
    Returns:
        是否所有路径都有效
    """
    if not image_path.exists():
        print(f"错误: 图像文件不存在: {image_path}")
        return False
    
    if xml_path and not xml_path.exists():
        print(f"错误: XML文件不存在: {xml_path}")
        return False
    
    return True


def determine_output_path(output_arg: Optional[str], image_path: Path) -> Path:
    """确定输出文件路径
    
    Args:
        output_arg: 命令行指定的输出路径
        image_path: 输入图像路径
        
    Returns:
        输出文件路径
    """
    if output_arg:
        return Path(output_arg)
    else:
        output_dir = Path('output/bccd_rf_detections')
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{image_path.stem}_detection.png"


# ============================================================================
# 9. 主程序相关函数
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
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
    
    return parser.parse_args()


def setup_device(device: str) -> str:
    """设置并验证计算设备
    
    Args:
        device: 期望的设备
        
    Returns:
        实际使用的设备
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        return 'cpu'
    return device


def print_header():
    """打印程序标题"""
    print("="*80)
    print("血细胞分类推理 - DINOv3特征 + 随机森林分类器")
    print("="*80)


def run_annotation_mode(image_path: Path, 
                       xml_path: Path,
                       dinov3_model: torch.nn.Module,
                       rf_classifier,
                       config: Dict,
                       device: str) -> Tuple[Image.Image, List[Dict]]:
    """运行标注框分类模式
    
    Args:
        image_path: 图像路径
        xml_path: XML标注路径
        dinov3_model: DINOv3模型
        rf_classifier: 随机森林分类器
        config: 配置字典
        device: 计算设备
        
    Returns:
        (可视化图像, 结果列表)
    """
    print(f"\n模式: 使用标注框进行分类")
    print(f"图像: {image_path}")
    print(f"标注: {xml_path}")
    
    vis_image, results = classify_with_annotations(
        image_path, xml_path, dinov3_model, rf_classifier, config, device
    )
    
    # 计算并打印统计信息
    stats = compute_classification_stats(results)
    per_class_stats = compute_per_class_stats(results, config['class_names'])
    print_classification_results(stats, per_class_stats, config['class_names'])
    
    return vis_image, results


def run_detection_mode(image_path: Path,
                      dinov3_model: torch.nn.Module,
                      rf_classifier,
                      config: Dict,
                      device: str,
                      conf_threshold: float,
                      nms_threshold: float) -> Tuple[Image.Image, List[Dict]]:
    """运行滑动窗口检测模式
    
    Args:
        image_path: 图像路径
        dinov3_model: DINOv3模型
        rf_classifier: 随机森林分类器
        config: 配置字典
        device: 计算设备
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
        
    Returns:
        (可视化图像, 结果列表)
    """
    print(f"\n模式: 滑动窗口检测")
    print(f"图像: {image_path}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"NMS阈值: {nms_threshold}")
    
    vis_image, results = detect_sliding_window(
        image_path, dinov3_model, rf_classifier, config,
        device, conf_threshold, nms_threshold
    )
    
    # 计算并打印统计信息
    stats = compute_detection_stats(results, config['class_names'])
    print_detection_results(stats)
    
    return vis_image, results


def main():
    """主函数：协调整个推理流程"""
    # 1. 解析参数和设置
    args = parse_arguments()
    args.device = setup_device(args.device)
    print_header()
    
    # 2. 加载模型和配置
    model_dir = Path(args.model_dir)
    dinov3_model, rf_classifier, config = load_model_and_config(
        model_dir, args.device, args.local_repo
    )
    
    # 3. 验证输入路径
    image_path = Path(args.image)
    xml_path = Path(args.xml) if args.xml else None
    
    if not validate_paths(image_path, xml_path):
        return
    
    # 4. 根据模式运行推理
    if xml_path:
        # 标注框分类模式
        vis_image, results = run_annotation_mode(
            image_path, xml_path, dinov3_model, rf_classifier, config, args.device
        )
    else:
        # 滑动窗口检测模式
        vis_image, results = run_detection_mode(
            image_path, dinov3_model, rf_classifier, config, 
            args.device, args.conf_threshold, args.nms_threshold
        )
    
    # 5. 保存结果
    output_path = determine_output_path(args.output, image_path)
    vis_image.save(output_path)
    print(f"\n✓ 结果保存到: {output_path}")
    
    print("="*80)
    print("推理完成！")
    print("="*80)


if __name__ == '__main__':
    main()
