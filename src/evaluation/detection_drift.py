"""
Detection decision-drift metrics between FP32 and quantized models.

This module compares two detection models on the same dataloader and computes:
  - miss_rate
  - hallucination_rate
  - class_flip_rate
  - score_drift
  - box_drift
  - gt_conditioned_miss_rate

Rates are returned in [0, 1].
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _is_yolo_like(model: Any) -> bool:
    """Best-effort check for our YOLO model wrappers."""
    if getattr(model, "_yolo", None) is not None:
        return True
    name = str(getattr(model, "name", "")).lower()
    return "yolo" in name


def _prepare_images_for_backbone(
    images: List[Any],
    is_yolo: bool,
    device: str,
) -> List[Any]:
    """
    Prepare a batch of images for a backbone.

    - YOLO wrappers accept numpy HWC (0-255) or tensors.
    - Torchvision detection models expect tensor CHW in [0, 1] on target device.
    """
    if is_yolo:
        return images

    prepared: List[torch.Tensor] = []
    for img in images:
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        elif isinstance(img, torch.Tensor):
            img_tensor = img
            # Convert HWC to CHW when needed.
            if img_tensor.dim() == 3 and img_tensor.shape[0] != 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            img_tensor = img_tensor.float()
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        prepared.append(img_tensor.to(device))

    return prepared


def _canonicalize_prediction(
    pred: Dict[str, Any],
    score_threshold: float,
    remap_yolo_labels_to_coco: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one detection prediction dict to (boxes, scores, labels) on CPU."""
    boxes = pred.get("boxes")
    scores = pred.get("scores")
    labels = pred.get("labels")

    if boxes is None or scores is None or labels is None:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )

    boxes = boxes.detach().cpu().float()
    scores = scores.detach().cpu().float()
    labels = labels.detach().cpu().long()
    if remap_yolo_labels_to_coco and labels.numel() > 0:
        # YOLO outputs 0..79 class ids. Targets use COCO category IDs.
        from src.evaluation.detection_metrics import remap_yolo_to_coco_labels
        labels = remap_yolo_to_coco_labels(labels).cpu().long()

    if boxes.numel() == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )

    keep = scores >= float(score_threshold)
    return boxes[keep], scores[keep], labels[keep]


def _canonicalize_target(
    target: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert one detection target dict to (gt_boxes, gt_labels) on CPU."""
    boxes = target.get("boxes")
    labels = target.get("labels")
    if boxes is None or labels is None:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )
    boxes = boxes.detach().cpu().float()
    labels = labels.detach().cpu().long()
    if boxes.numel() == 0 or labels.numel() == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )
    return boxes, labels


def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU matrix of shape [len(boxes1), len(boxes2)]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-9)


def _greedy_iou_matches(
    iou_matrix: torch.Tensor,
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    """
    Greedy one-to-one IoU matching.

    Returns list of tuples: (fp_idx, q_idx, iou_value).
    """
    if iou_matrix.numel() == 0:
        return []

    work = iou_matrix.clone()
    matches: List[Tuple[int, int, float]] = []

    n_cols = work.shape[1]
    while True:
        max_val = torch.max(work).item()
        if max_val < iou_threshold:
            break

        flat_idx = int(torch.argmax(work).item())
        i = flat_idx // n_cols
        j = flat_idx % n_cols
        matches.append((i, j, float(max_val)))

        # Remove matched row/col from further consideration.
        work[i, :] = -1.0
        work[:, j] = -1.0

    return matches


def _match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    """
    One-to-one class-aware matching from predictions to GT boxes.

    Returns tuples: (pred_idx, gt_idx, iou).
    """
    iou = _box_iou_matrix(pred_boxes, gt_boxes)
    if iou.numel() == 0:
        return []

    # Enforce class consistency for GT-conditioned matching.
    class_ok = pred_labels[:, None].eq(gt_labels[None, :])
    iou = torch.where(class_ok, iou, torch.full_like(iou, -1.0))
    return _greedy_iou_matches(iou, iou_threshold=iou_threshold)


def compute_detection_quantization_drift(
    fp32_model: Any,
    quantized_model: Any,
    dataloader: DataLoader,
    device: str,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
    gt_iou_threshold: float = 0.5,
    max_batches: Optional[int] = None,
    description: str = "Detection decision drift",
) -> Dict[str, float]:
    """
    Compare FP32 vs quantized detection outputs and compute drift metrics.

    Args:
        fp32_model: Model-like object with `backbone` and `name`.
        quantized_model: Model-like object with `backbone` and `name`.
        dataloader: Detection dataloader yielding (images, targets).
        device: Runtime device string.
        iou_threshold: IoU threshold used for detection matching.
        score_threshold: Confidence threshold applied to both models before matching.
        gt_iou_threshold: IoU threshold for class-aware prediction-vs-GT matching.
        max_batches: Optional hard cap on number of processed batches.
        description: tqdm label.

    Returns:
        Dictionary containing requested drift metrics in [0, 1] plus counts.
    """
    fp32_backbone = getattr(fp32_model, "backbone")
    quant_backbone = getattr(quantized_model, "backbone")
    fp32_is_yolo = _is_yolo_like(fp32_model)
    quant_is_yolo = _is_yolo_like(quantized_model)

    fp32_backbone.eval()
    quant_backbone.eval()

    total_fp = 0
    total_q = 0
    total_matches = 0
    total_miss = 0
    total_hallucination = 0
    total_class_flip = 0
    total_score_abs_diff = 0.0
    total_box_drift = 0.0
    total_fp_gt_tp = 0
    total_gt_conditioned_miss = 0
    total_images = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=description)
        for batch_idx, (images, targets) in enumerate(pbar):
            fp32_inputs = _prepare_images_for_backbone(images, fp32_is_yolo, device)
            quant_inputs = _prepare_images_for_backbone(images, quant_is_yolo, device)

            fp32_outputs = fp32_backbone(fp32_inputs)
            quant_outputs = quant_backbone(quant_inputs)

            for fp_pred, q_pred, target in zip(fp32_outputs, quant_outputs, targets):
                fp_boxes, fp_scores, fp_labels = _canonicalize_prediction(
                    fp_pred,
                    score_threshold,
                    remap_yolo_labels_to_coco=fp32_is_yolo,
                )
                q_boxes, q_scores, q_labels = _canonicalize_prediction(
                    q_pred,
                    score_threshold,
                    remap_yolo_labels_to_coco=quant_is_yolo,
                )
                gt_boxes, gt_labels = _canonicalize_target(target)

                n_fp = int(fp_boxes.shape[0])
                n_q = int(q_boxes.shape[0])
                total_fp += n_fp
                total_q += n_q
                total_images += 1

                iou_mat = _box_iou_matrix(fp_boxes, q_boxes)
                matches = _greedy_iou_matches(iou_mat, iou_threshold=iou_threshold)
                n_match = len(matches)
                total_matches += n_match

                total_miss += max(n_fp - n_match, 0)
                total_hallucination += max(n_q - n_match, 0)

                for fp_idx, q_idx, iou_value in matches:
                    if int(fp_labels[fp_idx].item()) != int(q_labels[q_idx].item()):
                        total_class_flip += 1
                    total_score_abs_diff += abs(
                        float(fp_scores[fp_idx].item()) - float(q_scores[q_idx].item())
                    )
                    total_box_drift += (1.0 - float(iou_value))

                # GT-conditioned miss:
                # Among GT objects detected by FP32 (class-aware TP), what fraction
                # are not detected by the quantized model.
                fp_gt_matches = _match_predictions_to_gt(
                    pred_boxes=fp_boxes,
                    pred_labels=fp_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    iou_threshold=gt_iou_threshold,
                )
                q_gt_matches = _match_predictions_to_gt(
                    pred_boxes=q_boxes,
                    pred_labels=q_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    iou_threshold=gt_iou_threshold,
                )
                fp_gt_detected = {gt_idx for _, gt_idx, _ in fp_gt_matches}
                q_gt_detected = {gt_idx for _, gt_idx, _ in q_gt_matches}
                total_fp_gt_tp += len(fp_gt_detected)
                total_gt_conditioned_miss += len(fp_gt_detected - q_gt_detected)

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    miss_rate = float(total_miss) / float(max(total_fp, 1))
    hallucination_rate = float(total_hallucination) / float(max(total_q, 1))
    class_flip_rate = float(total_class_flip) / float(max(total_matches, 1))
    score_drift = float(total_score_abs_diff) / float(max(total_matches, 1))
    box_drift = float(total_box_drift) / float(max(total_matches, 1))
    gt_conditioned_miss_rate = float(total_gt_conditioned_miss) / float(max(total_fp_gt_tp, 1))

    return {
        "miss_rate": miss_rate,
        "hallucination_rate": hallucination_rate,
        "class_flip_rate": class_flip_rate,
        "score_drift": score_drift,
        "box_drift": box_drift,
        "gt_conditioned_miss_rate": gt_conditioned_miss_rate,
        "total_images": float(total_images),
        "total_fp_detections": float(total_fp),
        "total_quantized_detections": float(total_q),
        "total_matches": float(total_matches),
        "total_fp_gt_true_positives": float(total_fp_gt_tp),
        "total_gt_conditioned_misses": float(total_gt_conditioned_miss),
    }
