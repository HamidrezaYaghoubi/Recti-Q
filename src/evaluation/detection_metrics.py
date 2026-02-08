"""
Detection metrics using pycocotools for proper COCO evaluation.

This module provides mAP computation following the official COCO evaluation protocol.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def compute_coco_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, Any]],
    coco_gt,
    iou_types: List[str] = ["bbox"],
) -> Dict[str, float]:
    """
    Compute COCO evaluation metrics using pycocotools.
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'.
        targets: List of target dicts with 'image_id'.
        coco_gt: COCO ground truth object from pycocotools.
        iou_types: Types of IoU evaluation (default: ["bbox"]).
        
    Returns:
        Dictionary with COCO metrics (mAP, mAP50, mAP75, etc.).
    """
    from pycocotools.cocoeval import COCOeval
    
    # Convert predictions to COCO format
    coco_results = []
    
    for pred, target in zip(predictions, targets):
        image_id = target["image_id"]
        if isinstance(image_id, torch.Tensor):
            image_id = image_id.item()
        
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        
        if len(boxes) == 0:
            continue
        
        # Convert boxes from [x1, y1, x2, y2] to [x, y, width, height]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        
        # COCO format: [x, y, width, height]
        boxes_coco = boxes.copy()
        boxes_coco[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_coco[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        
        for i in range(len(boxes)):
            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(labels[i]),
                "bbox": boxes_coco[i].tolist(),
                "score": float(scores[i]),
            })
    
    if len(coco_results) == 0:
        return {
            "mAP": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }
    
    # Load results into COCO format
    coco_dt = coco_gt.loadRes(coco_results)
    
    # Run evaluation
    metrics = {}
    
    for iou_type in iou_types:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        # stats order: [AP, AP50, AP75, AP_s, AP_m, AP_l, AR1, AR10, AR100, AR_s, AR_m, AR_l]
        stats = coco_eval.stats
        
        metrics.update({
            "mAP": float(stats[0]) * 100,        # AP @ IoU=0.50:0.95
            "mAP50": float(stats[1]) * 100,      # AP @ IoU=0.50
            "mAP75": float(stats[2]) * 100,      # AP @ IoU=0.75
            "mAP_small": float(stats[3]) * 100,  # AP for small objects
            "mAP_medium": float(stats[4]) * 100, # AP for medium objects
            "mAP_large": float(stats[5]) * 100,  # AP for large objects
            "AR_1": float(stats[6]) * 100,       # AR given 1 detection per image
            "AR_10": float(stats[7]) * 100,      # AR given 10 detections per image
            "AR_100": float(stats[8]) * 100,     # AR given 100 detections per image
            "AR_small": float(stats[9]) * 100,   # AR for small objects
            "AR_medium": float(stats[10]) * 100, # AR for medium objects
            "AR_large": float(stats[11]) * 100,  # AR for large objects
        })
    
    return metrics


class COCOEvaluator:
    """
    COCO-style evaluator for object detection.
    
    Accumulates predictions and computes metrics at the end.
    """
    
    def __init__(self, coco_gt, iou_types: List[str] = ["bbox"]):
        """
        Initialize COCO evaluator.
        
        Args:
            coco_gt: COCO ground truth object.
            iou_types: Types of IoU evaluation.
        """
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.predictions = []
        self.targets = []
    
    def reset(self):
        """Reset accumulated predictions."""
        self.predictions = []
        self.targets = []
    
    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, Any]],
    ):
        """
        Add batch of predictions and targets.
        
        Args:
            predictions: List of prediction dicts.
            targets: List of target dicts.
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute COCO metrics on accumulated predictions.
        
        Returns:
            Dictionary with COCO metrics.
        """
        return compute_coco_metrics(
            predictions=self.predictions,
            targets=self.targets,
            coco_gt=self.coco_gt,
            iou_types=self.iou_types,
        )


def remap_yolo_to_coco_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Remap YOLO class indices to COCO category IDs.
    
    YOLO uses 0-79 indices, but COCO uses specific category IDs (1-90 with gaps).
    
    Args:
        labels: Tensor of YOLO class indices (0-79).
        
    Returns:
        Tensor of COCO category IDs.
    """
    # COCO category IDs (80 categories, but IDs range from 1-90 with gaps)
    COCO_CATEGORY_IDS = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    ]
    
    # Create mapping tensor
    device = labels.device
    labels_np = labels.cpu().numpy()
    
    # Map each label
    remapped = np.array([COCO_CATEGORY_IDS[int(l)] for l in labels_np])
    
    return torch.from_numpy(remapped).to(device)
