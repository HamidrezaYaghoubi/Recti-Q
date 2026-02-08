"""
Evaluation module for computing metrics.
"""

from src.evaluation.metrics import (
    compute_accuracy,
    compute_top_k_accuracy,
    ClassificationMetrics,
    MetricsComputer,
)

from src.evaluation.detection_metrics import (
    compute_coco_metrics,
    COCOEvaluator,
    remap_yolo_to_coco_labels,
)

__all__ = [
    "compute_accuracy",
    "compute_top_k_accuracy",
    "ClassificationMetrics",
    "MetricsComputer",
    "compute_coco_metrics",
    "COCOEvaluator",
    "remap_yolo_to_coco_labels",
]

