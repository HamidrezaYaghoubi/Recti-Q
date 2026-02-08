"""
Evaluation metrics for classification and detection.

This module provides metrics computation for model evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from src.utils.logging import get_logger

logger = get_logger("qda.evaluation.metrics")


@dataclass
class ClassificationMetrics:
    """
    Container for classification metrics.
    
    Attributes:
        top1_accuracy: Top-1 accuracy (%).
        top5_accuracy: Top-5 accuracy (%).
        num_samples: Number of samples evaluated.
        per_class_accuracy: Optional per-class accuracy.
    """
    top1_accuracy: float
    top5_accuracy: float
    num_samples: int
    per_class_accuracy: Optional[Dict[int, float]] = None
    
    # Detailed metrics for analysis
    correct_indices: Optional[List[int]] = None
    incorrect_indices: Optional[List[int]] = None
    confidence_stats: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "top1_accuracy": self.top1_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "num_samples": self.num_samples,
        }
        if self.per_class_accuracy is not None:
            result["per_class_accuracy"] = self.per_class_accuracy
        if self.confidence_stats is not None:
            result["confidence_stats"] = self.confidence_stats
        return result
    
    def __str__(self) -> str:
        return (
            f"Top-1: {self.top1_accuracy:.2f}%, "
            f"Top-5: {self.top5_accuracy:.2f}%, "
            f"Samples: {self.num_samples}"
        )


def compute_accuracy(
    predictions: Tensor,
    labels: Tensor,
) -> float:
    """
    Compute top-1 accuracy.
    
    Args:
        predictions: Predicted class indices (N,).
        labels: Ground truth labels (N,).
        
    Returns:
        Accuracy as percentage (0-100).
    """
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}"
        )
    
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


def compute_top_k_accuracy(
    logits: Tensor,
    labels: Tensor,
    k: int = 5,
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model output logits (N, C).
        labels: Ground truth labels (N,).
        k: Number of top predictions to consider.
        
    Returns:
        Top-k accuracy as percentage (0-100).
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits, got shape {logits.shape}")
    
    batch_size = logits.size(0)
    
    # Get top-k predictions
    _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)
    
    # Check if true label is in top-k
    labels_expanded = labels.view(-1, 1).expand_as(top_k_preds)
    correct = top_k_preds.eq(labels_expanded).any(dim=1).sum().item()
    
    return 100.0 * correct / batch_size if batch_size > 0 else 0.0


class MetricsComputer:
    """
    Computes and accumulates metrics during evaluation.
    
    Supports incremental updates for batched evaluation.
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        track_per_class: bool = False,
        track_indices: bool = False,
    ):
        """
        Initialize metrics computer.
        
        Args:
            num_classes: Number of classes.
            track_per_class: Whether to track per-class metrics.
            track_indices: Whether to track correct/incorrect sample indices.
        """
        self.num_classes = num_classes
        self.track_per_class = track_per_class
        self.track_indices = track_indices
        
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.total_samples = 0
        self.top1_correct = 0
        self.top5_correct = 0
        
        # Per-class tracking
        if self.track_per_class:
            self.class_correct = np.zeros(self.num_classes)
            self.class_total = np.zeros(self.num_classes)
        
        # Index tracking
        if self.track_indices:
            self.correct_indices = []
            self.incorrect_indices = []
        
        # Confidence tracking
        self.confidences_correct = []
        self.confidences_incorrect = []
        
        # Store all predictions and labels
        self.all_predictions = []
        self.all_labels = []
        self.all_logits = []
    
    def update(
        self,
        logits: Tensor,
        labels: Tensor,
        indices: Optional[Tensor] = None,
    ) -> None:
        """
        Update metrics with a batch of predictions.
        
        Args:
            logits: Model output logits (N, C).
            labels: Ground truth labels (N,).
            indices: Optional sample indices for tracking.
        """
        batch_size = logits.size(0)
        self.total_samples += batch_size
        
        # Compute predictions and confidences
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        
        # Top-1 accuracy
        top1_correct_mask = predictions == labels
        self.top1_correct += top1_correct_mask.sum().item()
        
        # Top-5 accuracy
        _, top5_preds = logits.topk(5, dim=1, largest=True, sorted=True)
        labels_expanded = labels.view(-1, 1).expand_as(top5_preds)
        top5_correct_mask = top5_preds.eq(labels_expanded).any(dim=1)
        self.top5_correct += top5_correct_mask.sum().item()
        
        # Per-class accuracy
        if self.track_per_class:
            for i in range(batch_size):
                label = labels[i].item()
                self.class_total[label] += 1
                if top1_correct_mask[i]:
                    self.class_correct[label] += 1
        
        # Track indices (move mask to CPU to match indices)
        if self.track_indices and indices is not None:
            top1_correct_mask_cpu = top1_correct_mask.cpu()
            correct_idx = indices[top1_correct_mask_cpu].tolist()
            incorrect_idx = indices[~top1_correct_mask_cpu].tolist()
            self.correct_indices.extend(correct_idx)
            self.incorrect_indices.extend(incorrect_idx)
        
        # Track confidences
        correct_confs = confidences[top1_correct_mask].cpu().tolist()
        incorrect_confs = confidences[~top1_correct_mask].cpu().tolist()
        self.confidences_correct.extend(correct_confs)
        self.confidences_incorrect.extend(incorrect_confs)
        
        # Store all predictions
        self.all_predictions.extend(predictions.cpu().tolist())
        self.all_labels.extend(labels.cpu().tolist())
        self.all_logits.append(logits.cpu())
    
    def compute(self) -> ClassificationMetrics:
        """
        Compute final metrics.
        
        Returns:
            ClassificationMetrics object with all computed metrics.
        """
        top1_acc = 100.0 * self.top1_correct / self.total_samples if self.total_samples > 0 else 0.0
        top5_acc = 100.0 * self.top5_correct / self.total_samples if self.total_samples > 0 else 0.0
        
        # Per-class accuracy
        per_class_acc = None
        if self.track_per_class:
            per_class_acc = {}
            for i in range(self.num_classes):
                if self.class_total[i] > 0:
                    per_class_acc[i] = 100.0 * self.class_correct[i] / self.class_total[i]
        
        # Confidence statistics
        confidence_stats = None
        if self.confidences_correct or self.confidences_incorrect:
            all_confs = self.confidences_correct + self.confidences_incorrect
            confidence_stats = {
                "mean": np.mean(all_confs) if all_confs else 0.0,
                "std": np.std(all_confs) if all_confs else 0.0,
                "mean_correct": np.mean(self.confidences_correct) if self.confidences_correct else 0.0,
                "mean_incorrect": np.mean(self.confidences_incorrect) if self.confidences_incorrect else 0.0,
            }
        
        return ClassificationMetrics(
            top1_accuracy=top1_acc,
            top5_accuracy=top5_acc,
            num_samples=self.total_samples,
            per_class_accuracy=per_class_acc,
            correct_indices=self.correct_indices if self.track_indices else None,
            incorrect_indices=self.incorrect_indices if self.track_indices else None,
            confidence_stats=confidence_stats,
        )
    
    def get_all_predictions(self) -> Tuple[List[int], List[int], Tensor]:
        """
        Get all accumulated predictions.
        
        Returns:
            Tuple of (predictions, labels, logits).
        """
        all_logits = torch.cat(self.all_logits, dim=0) if self.all_logits else torch.tensor([])
        return self.all_predictions, self.all_labels, all_logits


def compute_decision_changes(
    predictions_fp32: Tensor,
    predictions_quantized: Tensor,
    labels: Tensor,
) -> Dict[str, Any]:
    """
    Compute decision changes between FP32 and quantized predictions.
    
    This is a key metric for analyzing quantization effects.
    
    Args:
        predictions_fp32: FP32 model predictions (N,).
        predictions_quantized: Quantized model predictions (N,).
        labels: Ground truth labels (N,).
        
    Returns:
        Dictionary with decision change statistics.
        
    TODO: Week 2-3 - Expand this analysis
    """
    # Decision changes
    changed = predictions_fp32 != predictions_quantized
    num_changed = changed.sum().item()
    
    # Categorize changes
    fp32_correct = predictions_fp32 == labels
    quant_correct = predictions_quantized == labels
    
    # Both correct (no functional change)
    both_correct = (fp32_correct & quant_correct).sum().item()
    
    # Both wrong (no functional change)
    both_wrong = (~fp32_correct & ~quant_correct).sum().item()
    
    # FP32 correct, quantized wrong (regression)
    regressions = (fp32_correct & ~quant_correct).sum().item()
    
    # FP32 wrong, quantized correct (improvement - rare)
    improvements = (~fp32_correct & quant_correct).sum().item()
    
    return {
        "total_samples": labels.size(0),
        "decision_changes": num_changed,
        "decision_change_rate": 100.0 * num_changed / labels.size(0),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "regressions": regressions,
        "improvements": improvements,
        "regression_rate": 100.0 * regressions / labels.size(0),
    }


# TODO: Week 2-4 - Additional metrics to implement:
#
# 1. Per-class decision change analysis
#    - Which classes are most affected by quantization?
#
# 2. Confidence-based analysis
#    - Do low-confidence predictions change more often?
#
# 3. Object size analysis (for detection)
#    - Do small objects suffer more from quantization?
#
# 4. Corruption robustness metrics
#    - mCE (mean Corruption Error)
#    - Relative mCE
#
# 5. Calibration metrics
#    - Expected Calibration Error (ECE)
#    - Maximum Calibration Error (MCE)
