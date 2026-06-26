"""
Main entry point for running experiments.

This module provides the main inference pipeline for:
- Loading models (FP32 baseline and quantized versions)
- Running inference on datasets
- Computing and saving metrics
- Logging to wandb

Usage:
    python -m src.main --config configs/baseline_classification.yaml
    python -m src.main --config configs/baseline_imagenet_c.yaml --debug
"""

import argparse
from dataclasses import replace
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, ExperimentConfig, ModelConfig
from src.utils.logging import setup_logging, get_logger, WandbLogger, MetricsLogger
from src.utils.checkpoint import CheckpointManager
from src.utils.formatting import (
    format_classification_results,
    format_detection_results,
    format_experiment_header,
    format_model_header,
    format_final_summary,
    format_quantization_stats,
    format_comparison_row,
)
from src.models import ModelFactory, BaseModel
from src.datasets import (
    get_imagenet_loader,
    get_imagenet_c_loader,
    get_all_imagenet_c_loaders,
    get_coco_loader,
    get_bdd100k_loader,
)
from src.evaluation import MetricsComputer, ClassificationMetrics
from src.quantization import quantize_model, QUANT_MODES, resolve_mode, get_model_size_mb


class QuantizationSkipped(RuntimeError):
    """Raised when a quantization run is skipped due to environment/backend limits."""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantization Decision Analysis - Inference Pipeline"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with subset of data",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)",
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to evaluate (overrides config)",
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific datasets to evaluate on (overrides config)",
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _safe_cuda_empty_cache(text_logger=None) -> None:
    """
    Best-effort CUDA cache cleanup.

    After a CUDA illegal-memory-access, empty_cache may throw again. This helper
    prevents those secondary failures from crashing the full experiment loop.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        if text_logger is not None:
            text_logger.warning(f"  [!] Skipping torch.cuda.empty_cache() due to CUDA state error: {e}")


def run_inference(
    model: BaseModel,
    dataloader: DataLoader,
    device: str,
    logger: MetricsLogger,
    description: str = "Inference",
) -> Dict[str, Any]:
    """
    Run inference on a dataset.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader for the dataset.
        device: Device to run on.
        logger: Metrics logger.
        description: Description for progress bar.
        
    Returns:
        Dictionary with predictions, labels, logits, and metrics.
    """
    model.eval()
    metrics_computer = MetricsComputer(
        num_classes=model.num_classes,
        track_per_class=True,
        track_indices=True,
    )
    
    all_predictions = []
    all_labels = []
    all_logits = []
    all_confidences = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=description)
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images = batch["image"]
                labels = batch["target"]
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            output = model.predict(images)
            
            # Update metrics
            indices = torch.arange(
                batch_idx * dataloader.batch_size,
                batch_idx * dataloader.batch_size + images.size(0)
            )
            metrics_computer.update(output.logits, labels, indices)
            
            # Store outputs
            all_predictions.append(output.predictions.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(output.logits.cpu())
            all_confidences.append(output.confidences.cpu())
            
            # Update progress bar
            current_metrics = metrics_computer.compute()
            pbar.set_postfix({
                "top1": f"{current_metrics.top1_accuracy:.2f}%",
                "top5": f"{current_metrics.top5_accuracy:.2f}%",
            })
    
    # Final metrics
    final_metrics = metrics_computer.compute()
    
    return {
        "predictions": torch.cat(all_predictions),
        "labels": torch.cat(all_labels),
        "logits": torch.cat(all_logits),
        "confidences": torch.cat(all_confidences),
        "metrics": final_metrics,
    }


def evaluate_model_on_imagenet(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
) -> ClassificationMetrics:
    """
    Evaluate a model on ImageNet validation set.
    
    Args:
        model: Model to evaluate.
        config: Experiment configuration.
        checkpoint_manager: Checkpoint manager for saving results.
        logger: Metrics logger.
        text_logger: Text logger.
        
    Returns:
        Classification metrics.
    """
    text_logger.info(f"  Evaluating on ImageNet...")
    
    # Get dataset config
    dataset_config = config.get_dataset("imagenet")
    
    # Create dataloader
    dataloader = get_imagenet_loader(
        config=dataset_config,
        model_name=model.name,
        num_workers=config.num_workers,
        debug=config.debug,
        debug_samples=config.debug_samples,
    )
    
    # Run inference
    results = run_inference(
        model=model,
        dataloader=dataloader,
        device=config.device,
        logger=logger,
        description=f"ImageNet - {model.name}",
    )
    
    # Log metrics
    metrics = results["metrics"]
    logger.log({
        f"{model.name}/imagenet/top1": metrics.top1_accuracy,
        f"{model.name}/imagenet/top5": metrics.top5_accuracy,
    })
    
    # Print formatted results
    formatted = format_classification_results(
        model_name=model.name,
        dataset_name="ImageNet",
        metrics=metrics,
        precision="fp32",
    )
    print("\n" + formatted)
    
    # Save predictions
    if config.output.save_predictions:
        save_data = {
            "predictions": results["predictions"].numpy(),
            "labels": results["labels"].numpy(),
            "confidences": results["confidences"].numpy(),
        }
        
        if config.output.save_logits:
            save_data["logits"] = results["logits"].numpy()
        
        checkpoint_manager.save_predictions(
            predictions=save_data,
            model_name=model.name,
            dataset_name="imagenet",
            precision="fp32",
        )
    
    # Save metrics
    checkpoint_manager.save_metrics(
        metrics=metrics.to_dict(),
        model_name=model.name,
        dataset_name="imagenet",
        precision="fp32",
    )
    
    return metrics


def evaluate_model_on_imagenet_c(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
) -> Dict[tuple, ClassificationMetrics]:
    """
    Evaluate a model on ImageNet-C.
    
    Args:
        model: Model to evaluate.
        config: Experiment configuration.
        checkpoint_manager: Checkpoint manager.
        logger: Metrics logger.
        text_logger: Text logger.
        
    Returns:
        Dictionary mapping (corruption, severity) to metrics.
    """
    text_logger.info(f"  Evaluating on ImageNet-C...")
    
    # Get dataset config
    dataset_config = config.get_dataset("imagenet_c")
    
    # Get all dataloaders
    loaders = get_all_imagenet_c_loaders(
        config=dataset_config,
        model_name=model.name,
        num_workers=config.num_workers,
    )
    
    all_metrics = {}
    
    for (corruption, severity), dataloader in loaders.items():
        # Optionally limit samples in debug mode
        if config.debug:
            from src.datasets.base import SubsetDataset
            dataset = SubsetDataset(dataloader.dataset, config.debug_samples)
            dataloader = DataLoader(
                dataset,
                batch_size=dataset_config.batch_size,
                num_workers=config.num_workers,
            )
        
        # Run inference
        results = run_inference(
            model=model,
            dataloader=dataloader,
            device=config.device,
            logger=logger,
            description=f"ImageNet-C {corruption}/s{severity}",
        )
        
        metrics = results["metrics"]
        all_metrics[(corruption, severity)] = metrics
        
        # Log metrics
        logger.log({
            f"{model.name}/imagenet_c/{corruption}/s{severity}/top1": metrics.top1_accuracy,
            f"{model.name}/imagenet_c/{corruption}/s{severity}/top5": metrics.top5_accuracy,
        })
        
        # Save predictions
        if config.output.save_predictions:
            save_data = {
                "predictions": results["predictions"].numpy(),
                "labels": results["labels"].numpy(),
                "confidences": results["confidences"].numpy(),
            }
            
            if config.output.save_logits:
                save_data["logits"] = results["logits"].numpy()
            
            checkpoint_manager.save_predictions(
                predictions=save_data,
                model_name=model.name,
                dataset_name=f"imagenet_c_{corruption}_s{severity}",
                precision="fp32",
            )
    
    # Compute mean metrics across corruptions
    mean_top1 = sum(m.top1_accuracy for m in all_metrics.values()) / len(all_metrics)
    mean_top5 = sum(m.top5_accuracy for m in all_metrics.values()) / len(all_metrics)
    
    # Print summary
    mean_metrics = ClassificationMetrics(top1_accuracy=mean_top1, top5_accuracy=mean_top5, num_samples=0, loss=0.0)
    formatted = format_classification_results(
        model_name=model.name,
        dataset_name=f"ImageNet-C ({len(all_metrics)} corruption combos)",
        metrics=mean_metrics,
        precision="fp32",
    )
    print("\n" + formatted)
    
    logger.log({
        f"{model.name}/imagenet_c/mean_top1": mean_top1,
        f"{model.name}/imagenet_c/mean_top5": mean_top5,
    })
    
    return all_metrics


def evaluate_model_on_coco(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
    precision: str = "fp32",
) -> Dict[str, Any]:
    """
    Evaluate a detection model on COCO dataset with proper mAP computation.
    
    Args:
        model: Model to evaluate.
        config: Experiment configuration.
        checkpoint_manager: Checkpoint manager.
        logger: Metrics logger.
        text_logger: Text logger.
        precision: Precision label for logging/saving (e.g. "fp32", "W8A8").
        
    Returns:
        Dictionary with detection metrics (mAP, mAP50, mAP75, etc.).
    """
    import sys
    import io
    from src.datasets import get_coco_loader
    from src.evaluation import COCOEvaluator, remap_yolo_to_coco_labels
    from pycocotools.coco import COCO
    
    text_logger.info(f"  Evaluating on COCO val2017 [{precision}]...")
    
    # Get dataset config
    dataset_config = config.get_dataset("coco")
    # Load COCO ground truth annotations (suppress verbose output)
    ann_file = Path(dataset_config.root) / "annotations" / "instances_val2017.json"
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_gt = COCO(str(ann_file))
    finally:
        sys.stdout = old_stdout
    
    # Create dataloader (use detection task)
    try:
        dataloader = get_coco_loader(
            config=dataset_config,
            task="detection",
            num_workers=config.num_workers,
            debug=config.debug,
            debug_samples=config.debug_samples,
        )
    except FileNotFoundError as e:
        text_logger.error(f"  [!] COCO dataset not found: {e}")
        text_logger.info("  Please download COCO dataset using: ./scripts/download_coco.sh")
        raise
    
    model.eval()
    device = config.device
    
    # Initialize COCO evaluator
    coco_evaluator = COCOEvaluator(coco_gt)
    
    # Check if model is YOLO (needs special handling)
    is_yolo = "yolo" in model.name.lower()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"COCO - {model.name}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Handle images based on model type
            # Detection dataloader returns numpy arrays (HWC, 0-255)
            if is_yolo:
                # YOLO expects numpy arrays, keep as-is
                processed_images = images
            else:
                # Torchvision detection models expect tensors
                import numpy as np
                processed_images = []
                for img in images:
                    if isinstance(img, np.ndarray):
                        # Convert numpy HWC to tensor CHW and normalize to [0, 1]
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        processed_images.append(img_tensor.to(device))
                    else:
                        processed_images.append(img.to(device))
            
            # Get predictions
            outputs = model.backbone(processed_images)
            
            # Store outputs and update evaluator
            batch_preds = []
            for i, (output, target) in enumerate(zip(outputs, targets)):
                labels = output["labels"].cpu()
                
                # Remap YOLO labels (0-79) to COCO category IDs (1-90 with gaps)
                if is_yolo:
                    labels = remap_yolo_to_coco_labels(labels)
                
                pred = {
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": labels,
                }
                batch_preds.append(pred)
                all_predictions.append(pred)
                all_targets.append(target)
            
            # Update evaluator
            coco_evaluator.update(batch_preds, targets)
            
            if config.debug and batch_idx >= config.debug_samples // dataset_config.batch_size:
                break
    
    # Compute COCO metrics (mAP)
    metrics = coco_evaluator.compute()
    
    # Add summary stats
    total_detections = sum(len(p["boxes"]) for p in all_predictions)
    metrics["num_images"] = len(all_predictions)
    metrics["total_detections"] = total_detections
    metrics["avg_detections_per_image"] = total_detections / len(all_predictions) if all_predictions else 0
    
    # Print formatted results
    formatted = format_detection_results(
        model_name=model.name,
        dataset_name="COCO val2017",
        metrics=metrics,
        precision=precision,
        verbose=True,  # Show size breakdown
    )
    print("\n" + formatted)

    # Log flat keys for clean cross-run comparison in wandb.
    detection_log_metrics = {
        "map": metrics["mAP"],
        "map_50": metrics["mAP50"],
        "map75": metrics["mAP75"],
        "map_small": metrics["mAP_small"],
        "map_medium": metrics["mAP_medium"],
        "map_large": metrics["mAP_large"],
        "ar_100": metrics["AR_100"],
    }
    logger.log(detection_log_metrics)

    # Also write concise summary keys for run-table comparison.
    if getattr(logger, "wandb_logger", None) is not None:
        logger.wandb_logger.log_summary(
            {
                "map": metrics["mAP"],
                "map_50": metrics["mAP50"],
                "map75": metrics["mAP75"],
                "map_small": metrics["mAP_small"],
                "map_medium": metrics["mAP_medium"],
                "map_large": metrics["mAP_large"],
                "ar_100": metrics["AR_100"],
            }
        )
    
    # Save predictions
    if config.output.save_predictions:
        checkpoint_manager.save_predictions(
            predictions={"predictions": all_predictions, "targets": all_targets, "metrics": metrics},
            model_name=model.name,
            dataset_name="coco",
            precision=precision,
        )
    
    # Save metrics
    checkpoint_manager.save_metrics(
        metrics=metrics,
        model_name=model.name,
        dataset_name="coco",
        precision=precision,
    )
    
    return metrics


def _extract_ultralytics_detection_metrics(val_results) -> Dict[str, float]:
    """
    Extract detection metrics from ultralytics val() results.
    """
    box = getattr(val_results, "box", None)
    results_dict = getattr(val_results, "results_dict", {}) or {}

    def _from_box(attr: str, fallback_key: str) -> float:
        if box is not None and hasattr(box, attr):
            try:
                return float(getattr(box, attr)) * 100.0
            except Exception:
                pass
        return float(results_dict.get(fallback_key, 0.0)) * 100.0

    return {
        "mAP": _from_box("map", "metrics/mAP50-95(B)"),
        "mAP50": _from_box("map50", "metrics/mAP50(B)"),
        "mAP75": _from_box("map75", "metrics/mAP75(B)"),
        # Not always available from ultralytics val() summaries on custom YOLO sets.
        "mAP_small": 0.0,
        "mAP_medium": 0.0,
        "mAP_large": 0.0,
        "AR_100": 0.0,
    }


def _get_yolo_runtime_handle(model: BaseModel):
    """
    Return the underlying ultralytics YOLO runtime object if available.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "yolo"):
        return backbone.yolo
    yolo_obj = getattr(model, "_yolo", None)
    if yolo_obj is not None:
        return yolo_obj
    return None


def _build_bdd_data_yaml(config: ExperimentConfig) -> Path:
    """
    Build a local BDD100K YOLO dataset yaml.

    This ignores upstream absolute Kaggle paths and uses local root from config.
    """
    import yaml

    dataset_cfg = config.get_dataset("bdd100k")
    root = Path(dataset_cfg.root).resolve()

    if not (root / "train" / "images").exists():
        raise FileNotFoundError(f"Expected BDD100K train images at: {root / 'train' / 'images'}")
    if not (root / "val" / "images").exists():
        raise FileNotFoundError(f"Expected BDD100K val images at: {root / 'val' / 'images'}")

    names = [
        "person",
        "rider",
        "car",
        "bus",
        "truck",
        "bike",
        "motor",
        "traffic light",
        "traffic sign",
        "train",
    ]
    # Try to preserve names from dataset data.yaml if present.
    source_yaml = root / "data.yaml"
    if source_yaml.exists():
        try:
            with open(source_yaml, "r") as f:
                payload = yaml.safe_load(f) or {}
            yaml_names = payload.get("names")
            if isinstance(yaml_names, list) and yaml_names:
                names = [str(x) for x in yaml_names]
            elif isinstance(yaml_names, dict) and yaml_names:
                names = [str(yaml_names[k]) for k in sorted(yaml_names.keys(), key=lambda v: int(v))]
        except Exception:
            pass

    data_yaml = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }

    out_dir = Path(config.output.results_dir) / config.name / "yolo_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bdd100k_yolo_eval.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return out_path


def evaluate_model_on_bdd(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
    precision: str = "fp32",
) -> Dict[str, Any]:
    """
    Evaluate YOLO detection model on BDD100K using ultralytics val().
    """
    dataset_config = config.get_dataset("bdd100k")
    split = dataset_config.split or "val"

    yolo_runtime = _get_yolo_runtime_handle(model)
    if yolo_runtime is None:
        raise QuantizationSkipped(
            "BDD100K evaluation currently supports YOLO-backed detection models only."
        )

    data_arg = config.quantization.yolo_data or str(_build_bdd_data_yaml(config))
    text_logger.info(f"  Evaluating on BDD100K [{precision}]...")

    eval_batch = int(dataset_config.batch_size)
    if "ENGINE" in precision.upper():
        trt_max_batch = int(config.quantization.yolo_batch)
        if eval_batch > trt_max_batch:
            text_logger.info(
                f"  BDD eval batch={eval_batch} exceeds TensorRT profile max batch={trt_max_batch}; "
                f"using batch={trt_max_batch}."
            )
            eval_batch = trt_max_batch

    val_results = yolo_runtime.val(
        data=data_arg,
        split=split,
        batch=eval_batch,
        imgsz=config.quantization.yolo_imgsz,
        device=config.device,
        verbose=False,
    )

    metrics = _extract_ultralytics_detection_metrics(val_results)
    metrics["num_images"] = 0
    metrics["total_detections"] = 0
    metrics["avg_detections_per_image"] = 0.0

    formatted = format_detection_results(
        model_name=model.name,
        dataset_name=f"BDD100K {split}",
        metrics=metrics,
        precision=precision,
        verbose=True,
    )
    print("\n" + formatted)

    logger.log(
        {
            "map": metrics["mAP"],
            "map_50": metrics["mAP50"],
            "map75": metrics["mAP75"],
            "map_small": metrics["mAP_small"],
            "map_medium": metrics["mAP_medium"],
            "map_large": metrics["mAP_large"],
            "ar_100": metrics["AR_100"],
        }
    )

    if getattr(logger, "wandb_logger", None) is not None:
        logger.wandb_logger.log_summary(
            {
                "map": metrics["mAP"],
                "map_50": metrics["mAP50"],
                "map75": metrics["mAP75"],
                "map_small": metrics["mAP_small"],
                "map_medium": metrics["mAP_medium"],
                "map_large": metrics["mAP_large"],
                "ar_100": metrics["AR_100"],
            }
        )

    checkpoint_manager.save_metrics(
        metrics=metrics,
        model_name=model.name,
        dataset_name="bdd100k",
        precision=precision,
    )

    return metrics


def evaluate_model_on_detection_dataset(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
    dataset_name: str,
    precision: str = "fp32",
) -> Dict[str, Any]:
    """
    Dispatch detection evaluation to the selected dataset.
    """
    if dataset_name == "coco":
        return evaluate_model_on_coco(
            model=model,
            config=config,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            text_logger=text_logger,
            precision=precision,
        )
    if dataset_name == "bdd100k":
        return evaluate_model_on_bdd(
            model=model,
            config=config,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            text_logger=text_logger,
            precision=precision,
        )
    raise ValueError(f"Unsupported detection dataset '{dataset_name}'")


# ========================================================================
# Quantized evaluation helpers
# ========================================================================

def evaluate_detection_quantization_drift(
    fp32_model: Any,
    quantized_model: Any,
    config: ExperimentConfig,
    logger: MetricsLogger,
    text_logger,
    precision_label: str,
    dataset_name: str,
) -> Dict[str, float]:
    """
    Compute FP32-vs-quantized detection drift on a detection dataset and log to wandb.

    This is an output-level comparison (decision drift), not mAP.
    """
    from src.datasets import get_coco_loader, get_bdd100k_loader
    from src.evaluation import compute_detection_quantization_drift

    if dataset_name == "coco":
        dataset_config = config.get_dataset("coco")
        drift_batch_size = int(dataset_config.batch_size)
        yolo_batch = int(getattr(config.quantization, "yolo_batch", 0) or 0)
        if yolo_batch > 0:
            drift_batch_size = min(drift_batch_size, yolo_batch)
        drift_dataset_config = replace(
            dataset_config,
            batch_size=drift_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        dataloader = get_coco_loader(
            config=drift_dataset_config,
            task="detection",
            num_workers=0,
            debug=config.debug,
            debug_samples=config.debug_samples,
        )
        remap_yolo_labels_to_coco = True
    elif dataset_name == "bdd100k":
        dataset_config = config.get_dataset("bdd100k")
        drift_batch_size = int(dataset_config.batch_size)
        yolo_batch = int(getattr(config.quantization, "yolo_batch", 0) or 0)
        if yolo_batch > 0:
            drift_batch_size = min(drift_batch_size, yolo_batch)
        drift_dataset_config = replace(
            dataset_config,
            batch_size=drift_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        dataloader = get_bdd100k_loader(
            config=drift_dataset_config,
            task="detection",
            num_workers=0,
            debug=config.debug,
            debug_samples=config.debug_samples,
        )
        remap_yolo_labels_to_coco = False
    else:
        raise QuantizationSkipped(
            f"Decision drift is not implemented for detection dataset '{dataset_name}'."
        )

    model_name = str(getattr(fp32_model, "name", "model"))
    text_logger.info(
        "  Computing decision drift on "
        f"{dataset_name.upper()} [{precision_label}] "
        f"(num_workers=0, pin_memory=false, batch_size={drift_batch_size})..."
    )
    drift_all = compute_detection_quantization_drift(
        fp32_model=fp32_model,
        quantized_model=quantized_model,
        dataloader=dataloader,
        device=config.device,
        remap_yolo_labels_to_coco=remap_yolo_labels_to_coco,
        description=f"Drift - {model_name} [{precision_label}]",
    )

    keys = [
        "miss_rate",
        "hallucination_rate",
        "class_flip_rate",
        "score_drift",
        "box_drift",
        "gt_conditioned_miss_rate",
    ]
    drift_metrics = {k: float(drift_all[k]) for k in keys}

    logger.log({f"drift_{k}": v for k, v in drift_metrics.items()})

    if getattr(logger, "wandb_logger", None) is not None:
        logger.wandb_logger.log_summary({f"drift_{k}": v for k, v in drift_metrics.items()})

    text_logger.info(
        "  Decision drift summary: "
        f"miss={drift_metrics['miss_rate']:.4f}, "
        f"hallucination={drift_metrics['hallucination_rate']:.4f}, "
        f"class_flip={drift_metrics['class_flip_rate']:.4f}, "
        f"score_drift={drift_metrics['score_drift']:.4f}, "
        f"box_drift={drift_metrics['box_drift']:.4f}, "
        f"gt_cond_miss={drift_metrics['gt_conditioned_miss_rate']:.4f}"
    )

    return drift_metrics


def evaluate_rectiq_student_closeness(
    runtime_export_model: Any,
    ptq_surrogate_model: Any,
    config: ExperimentConfig,
    logger: MetricsLogger,
    text_logger,
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    max_batches: Optional[int],
    tag: str,
) -> Dict[str, float]:
    """
    Compare runtime-export INT8 vs PTQ-surrogate student before Recti-Q training.
    """
    from src.evaluation import compute_detection_quantization_drift

    dataloader = _build_detection_loader_for_split(
        config=config,
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        num_workers=max(int(num_workers), 0),
        shuffle=False,
    )
    remap_yolo_labels_to_coco = dataset_name == "coco"

    text_logger.info(
        "  Computing PTQ-student closeness vs runtime-export: "
        f"dataset={dataset_name}, split={split}, batch={batch_size}, "
        f"max_batches={max_batches}, tag={tag}"
    )
    closeness_all = compute_detection_quantization_drift(
        fp32_model=runtime_export_model,
        quantized_model=ptq_surrogate_model,
        dataloader=dataloader,
        device=config.device,
        remap_yolo_labels_to_coco=remap_yolo_labels_to_coco,
        max_batches=max_batches,
        description=f"Student closeness - {tag}",
    )

    keys = [
        "miss_rate",
        "hallucination_rate",
        "class_flip_rate",
        "score_drift",
        "box_drift",
        "gt_conditioned_miss_rate",
        "total_images",
        "total_fp_detections",
        "total_quantized_detections",
        "total_matches",
        "total_fp_gt_true_positives",
        "total_gt_conditioned_misses",
    ]
    closeness_metrics = {k: float(closeness_all[k]) for k in keys if k in closeness_all}

    logger.log({f"student_closeness_{k}": v for k, v in closeness_metrics.items()})
    if getattr(logger, "wandb_logger", None) is not None:
        logger.wandb_logger.log_summary(
            {f"student_closeness_{k}": v for k, v in closeness_metrics.items()}
        )

    text_logger.info(
        "  PTQ-student closeness summary: "
        f"miss={closeness_metrics.get('miss_rate', 0.0):.4f}, "
        f"hallucination={closeness_metrics.get('hallucination_rate', 0.0):.4f}, "
        f"class_flip={closeness_metrics.get('class_flip_rate', 0.0):.4f}, "
        f"score_drift={closeness_metrics.get('score_drift', 0.0):.4f}, "
        f"box_drift={closeness_metrics.get('box_drift', 0.0):.4f}, "
        f"gt_cond_miss={closeness_metrics.get('gt_conditioned_miss_rate', 0.0):.4f}"
    )

    return closeness_metrics


def _default_detection_split(dataset_name: str, phase: str) -> str:
    """
    Return default split names for detection datasets.
    """
    if dataset_name == "coco":
        return "train2017" if phase == "train" else "val2017"
    if dataset_name == "bdd100k":
        return "train" if phase == "train" else "val"
    return "val"


def _build_detection_loader_for_split(
    config: ExperimentConfig,
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    """
    Build a detection dataloader with explicit split/batch overrides.
    """
    dataset_cfg = config.get_dataset(dataset_name)
    loader_cfg = replace(
        dataset_cfg,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    if dataset_name == "coco":
        return get_coco_loader(
            config=loader_cfg,
            task="detection",
            num_workers=num_workers,
            debug=config.debug,
            debug_samples=config.debug_samples,
        )
    if dataset_name == "bdd100k":
        return get_bdd100k_loader(
            config=loader_cfg,
            task="detection",
            num_workers=num_workers,
            debug=config.debug,
            debug_samples=config.debug_samples,
        )
    raise QuantizationSkipped(
        f"Recti-Q is not implemented for detection dataset '{dataset_name}'."
    )


def evaluate_rectiq_yolo_detection(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
    dataset_name: str,
    base_precision_label: str,
    quant_stats: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Train and evaluate Recti-Q adapter on top of a quantized YOLO student.
    """
    from ultralytics import YOLO

    from src.models.detection import YOLOWrapper
    from src.rectiq import (
        RectiQConfig as RectiQTrainConfig,
        attach_fixed_int8_detect_input_quantizer,
        calibrate_detect_input_ptq_scales,
        save_rectiq_adapter,
        train_rectiq_adapter,
    )

    if not _is_yolo_model(model):
        raise QuantizationSkipped("Recti-Q is currently wired for YOLO-backed detection models only.")
    if not config.rectiq.enabled:
        raise QuantizationSkipped("Recti-Q is disabled in config (set rectiq.enabled=true).")

    rectiq_cfg = config.rectiq
    student_backend = str(getattr(rectiq_cfg, "student_backend", "ptq_surrogate")).strip().lower()
    if student_backend in {"ptq", "pytorch_ptq"}:
        student_backend = "ptq_surrogate"
    if student_backend in {"runtime", "export"}:
        student_backend = "runtime_export"
    if student_backend not in {"ptq_surrogate", "runtime_export"}:
        raise QuantizationSkipped(
            f"Unknown Recti-Q student backend '{student_backend}'. "
            "Use one of: ptq_surrogate, runtime_export."
        )

    teacher_for_rectiq = None
    if rectiq_cfg.use_teacher and float(rectiq_cfg.feature_kd_weight) > 0.0:
        teacher_for_rectiq = model
    if teacher_for_rectiq is None and float(rectiq_cfg.task_loss_weight) <= 0.0:
        raise QuantizationSkipped(
            "Recti-Q has no active training signal. "
            "Set rectiq.task_loss_weight > 0 for detection supervision "
            "and/or enable teacher KD (rectiq.use_teacher=true, rectiq.feature_kd_weight>0)."
        )

    dataset_cfg = config.get_dataset(dataset_name)
    train_split = rectiq_cfg.train_split or _default_detection_split(dataset_name, "train")
    val_split = (
        rectiq_cfg.val_split
        or dataset_cfg.split
        or _default_detection_split(dataset_name, "val")
    )
    train_batch_size = int(rectiq_cfg.train_batch_size or dataset_cfg.batch_size)
    val_batch_size = int(rectiq_cfg.val_batch_size or dataset_cfg.batch_size)
    train_num_workers = max(int(rectiq_cfg.num_workers), 0)
    train_imgsz = int(rectiq_cfg.imgsz or config.quantization.yolo_imgsz)

    text_logger.info(
        "  Recti-Q setup: "
        f"dataset={dataset_name}, train_split={train_split}, val_split={val_split}, "
        f"student_backend={student_backend}, "
        f"adapter_target={str(getattr(rectiq_cfg, 'adapter_target', 'detect_input')).strip().lower()}, "
        f"epochs={rectiq_cfg.epochs}, rank={rectiq_cfg.rank}, alpha={rectiq_cfg.alpha}, "
        f"batch(train/val)=({train_batch_size}/{val_batch_size}), workers={train_num_workers}, "
        f"feature_kd_weight={rectiq_cfg.feature_kd_weight}, task_loss_weight={rectiq_cfg.task_loss_weight}, "
        f"use_teacher={bool(teacher_for_rectiq is not None)}, "
        f"use_dwconv={bool(getattr(rectiq_cfg, 'adapter_use_dwconv', True))}, "
        f"dropout={float(getattr(rectiq_cfg, 'adapter_dropout', 0.0))}, "
        f"requantize_after_residual={bool(getattr(rectiq_cfg, 'requantize_after_residual', True))}, "
        f"unfreeze_cv3={bool(getattr(rectiq_cfg, 'unfreeze_detect_cv3', False))}"
    )

    source_loader = _build_detection_loader_for_split(
        config=config,
        dataset_name=dataset_name,
        split=train_split,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        shuffle=True,
    )
    val_loader = _build_detection_loader_for_split(
        config=config,
        dataset_name=dataset_name,
        split=val_split,
        batch_size=val_batch_size,
        num_workers=train_num_workers,
        shuffle=False,
    )

    export_path_raw = quant_stats.get("export_path")
    export_path: Optional[Path] = None
    if export_path_raw:
        export_path = Path(export_path_raw).resolve()

    ptq_scales: Optional[List[float]] = None
    ptq_calibration_stats: Dict[str, Any] = {}
    student_closeness_stats: Dict[str, Any] = {}

    if student_backend == "runtime_export":
        if export_path is None:
            raise QuantizationSkipped(
                "Recti-Q runtime_export backend requires quantized export_path, but none was found."
            )
        if not export_path.exists():
            raise QuantizationSkipped(
                f"Recti-Q export path does not exist: {export_path}"
            )

        quantized_yolo = YOLO(str(export_path), task="detect")
        if not isinstance(getattr(quantized_yolo, "model", None), torch.nn.Module):
            raise QuantizationSkipped(
                "Recti-Q runtime_export backend is not trainable for this export. "
                f"Export '{export_path}' uses inference-only runtime backend "
                f"(model type: {type(getattr(quantized_yolo, 'model', None))}). "
                "Use rectiq.student_backend='ptq_surrogate' for PTQ-style trainable student."
            )
        quantized_yolo.overrides.update(
            {
                "task": "detect",
                "mode": "predict",
                "data": None,
                "verbose": False,
            }
        )
        quantized_backbone = YOLOWrapper(quantized_yolo)
    else:
        # PTQ-trainable surrogate student:
        # load the same source checkpoint as FP32 model, then attach fixed INT8 Q/DQ
        # at Detect input features to emulate frozen PTQ behavior with gradients.
        source_weights = _infer_yolo_source_weights_id(model)
        if source_weights.startswith("builtin:"):
            source_weights = f"{source_weights.split(':', 1)[1]}.pt"
        quantized_yolo = YOLO(str(source_weights), task="detect")
        if not isinstance(getattr(quantized_yolo, "model", None), torch.nn.Module):
            raise QuantizationSkipped(
                f"Recti-Q ptq_surrogate backend expected PyTorch YOLO model, got {type(getattr(quantized_yolo, 'model', None))}."
            )
        # Keep surrogate backend on the configured device for calibration/training.
        quantized_yolo.model = quantized_yolo.model.to(config.device)
        quantized_yolo.overrides.update(
            {
                "task": "detect",
                "mode": "predict",
                "data": None,
                "verbose": False,
            }
        )
        quantized_backbone = YOLOWrapper(quantized_yolo)

        ptq_calib_batches = (
            rectiq_cfg.ptq_calibration_batches
            if rectiq_cfg.ptq_calibration_batches is not None
            else config.quantization.num_calibration_batches
        )
        if ptq_calib_batches is None:
            ptq_calib_batches = 50
        ptq_calibration_stats = calibrate_detect_input_ptq_scales(
            model_like=quantized_backbone,
            source_loader=source_loader,
            device=config.device,
            imgsz=train_imgsz,
            max_batches=int(ptq_calib_batches),
        )
        ptq_scales = list(ptq_calibration_stats.get("scales", []))
        text_logger.info(
            "  Recti-Q PTQ surrogate calibrated: "
            f"features={ptq_calibration_stats.get('num_features', 0)}, "
            f"batches={ptq_calibration_stats.get('num_batches', 0)}, "
            f"use_ste={bool(rectiq_cfg.ptq_use_ste)}"
        )

        if bool(rectiq_cfg.compare_to_runtime_export):
            if export_path is None or not export_path.exists():
                text_logger.warning(
                    "  [!] Skipping PTQ-student closeness report: runtime export_path is unavailable."
                )
            else:
                temp_quantizer = None
                try:
                    # For closeness diagnostics only, emulate the PTQ surrogate
                    # without Recti-Q residuals by attaching fixed Q/DQ.
                    temp_quantizer = attach_fixed_int8_detect_input_quantizer(
                        model_like=quantized_backbone,
                        scales=ptq_scales,
                        use_ste=bool(rectiq_cfg.ptq_use_ste),
                    )
                    runtime_export_yolo = YOLO(str(export_path), task="detect")
                    runtime_export_yolo.overrides.update(
                        {
                            "task": "detect",
                            "mode": "predict",
                            "data": None,
                            "verbose": False,
                        }
                    )
                    runtime_export_backbone = YOLOWrapper(runtime_export_yolo)
                    runtime_view = SimpleNamespace(
                        name=f"{model.name}_runtime_export",
                        backbone=runtime_export_backbone,
                        _yolo=getattr(model, "_yolo", None),
                    )
                    surrogate_view = SimpleNamespace(
                        name=f"{model.name}_ptq_surrogate",
                        backbone=quantized_backbone,
                        _yolo=getattr(model, "_yolo", None),
                    )
                    student_closeness_stats = evaluate_rectiq_student_closeness(
                        runtime_export_model=runtime_view,
                        ptq_surrogate_model=surrogate_view,
                        config=config,
                        logger=logger,
                        text_logger=text_logger,
                        dataset_name=dataset_name,
                        split=val_split,
                        batch_size=val_batch_size,
                        num_workers=train_num_workers,
                        max_batches=rectiq_cfg.compare_max_batches,
                        tag=f"{model.name}:{dataset_name}:{base_precision_label}",
                    )
                except Exception as e:
                    text_logger.warning(
                        f"  [!] PTQ-student closeness report failed for {model.name}: {e}"
                    )
                finally:
                    if temp_quantizer is not None:
                        temp_quantizer.remove()

    rectiq_train_cfg = RectiQTrainConfig(
        rank=int(rectiq_cfg.rank),
        rank_per_scale=(
            [int(v) for v in rectiq_cfg.rank_per_scale]
            if getattr(rectiq_cfg, "rank_per_scale", None)
            else None
        ),
        alpha=float(rectiq_cfg.alpha),
        alpha_per_scale=(
            [float(v) for v in rectiq_cfg.alpha_per_scale]
            if getattr(rectiq_cfg, "alpha_per_scale", None)
            else None
        ),
        imgsz=train_imgsz,
        epochs=int(rectiq_cfg.epochs),
        lr=float(rectiq_cfg.lr),
        weight_decay=float(rectiq_cfg.weight_decay),
        feature_kd_weight=float(rectiq_cfg.feature_kd_weight),
        residual_reg_weight=float(rectiq_cfg.residual_reg_weight),
        task_loss_weight=float(rectiq_cfg.task_loss_weight),
        adapter_target=str(getattr(rectiq_cfg, "adapter_target", "detect_input")),
        adapter_use_dwconv=bool(getattr(rectiq_cfg, "adapter_use_dwconv", True)),
        adapter_dropout=float(getattr(rectiq_cfg, "adapter_dropout", 0.0)),
        requantize_after_residual=bool(getattr(rectiq_cfg, "requantize_after_residual", True)),
        ptq_scales=(ptq_scales if student_backend == "ptq_surrogate" else None),
        ptq_use_ste=bool(rectiq_cfg.ptq_use_ste),
        unfreeze_detect_cv3=bool(getattr(rectiq_cfg, "unfreeze_detect_cv3", False)),
        cv3_lr=(
            float(rectiq_cfg.cv3_lr)
            if getattr(rectiq_cfg, "cv3_lr", None) is not None
            else None
        ),
        recalibration_epoch=(
            int(rectiq_cfg.recalibration_epoch)
            if getattr(rectiq_cfg, "recalibration_epoch", None) is not None
            else None
        ),
        recalibration_batches=(
            int(rectiq_cfg.recalibration_batches)
            if getattr(rectiq_cfg, "recalibration_batches", None) is not None
            else None
        ),
        max_batches_per_epoch=(
            int(rectiq_cfg.max_batches_per_epoch)
            if rectiq_cfg.max_batches_per_epoch is not None
            else None
        ),
        val_final_only=bool(rectiq_cfg.val_final_only),
    )

    try:
        t0 = time.time()
        try:
            rectiq_result = train_rectiq_adapter(
                quantized_model=quantized_backbone,
                source_loader=source_loader,
                device=config.device,
                config=rectiq_train_cfg,
                teacher_model=teacher_for_rectiq,
                val_loader=val_loader,
                task_loss_fn=None,
            )
        except Exception as e:
            raise QuantizationSkipped(
                f"Recti-Q training failed for {model.name} [{base_precision_label}] on {dataset_name}: {e}"
            ) from e
        rectiq_time = time.time() - t0

        adapter_out_dir = Path(
            rectiq_cfg.output_dir or (Path(config.output.results_dir) / config.name / "rectiq_adapters")
        )
        adapter_out_dir.mkdir(parents=True, exist_ok=True)
        safe_precision = base_precision_label.lower().replace("/", "_")
        adapter_path = adapter_out_dir / f"{model.name}_{dataset_name}_{safe_precision}_rectiq.pt"
        save_rectiq_adapter(
            adapter=rectiq_result.adapter,
            save_path=adapter_path,
            extra={
                "model_name": model.name,
                "dataset_name": dataset_name,
                "base_precision_label": base_precision_label,
                "train_split": train_split,
                "val_split": val_split,
                "best_epoch": int(rectiq_result.best_epoch),
                "best_val_loss": float(rectiq_result.best_val_loss),
                "rectiq_time_s": float(rectiq_time),
                "seed": int(config.seed),
                "student_backend": student_backend,
                "ptq_calibration": ptq_calibration_stats,
            },
        )

        adapter_size_mb = adapter_path.stat().st_size / (1024 ** 2)
        fallback_student_size = get_model_size_mb(quantized_yolo.model)
        base_quantized_size_mb = float(
            quant_stats.get("quantized_size_mb", quant_stats.get("original_size_mb", fallback_student_size))
        )
        effective_quantized_size_mb = base_quantized_size_mb + adapter_size_mb
        original_size_mb = float(quant_stats.get("original_size_mb", effective_quantized_size_mb))
        compression_ratio = original_size_mb / max(effective_quantized_size_mb, 1e-6)
        size_reduction_pct = (1 - effective_quantized_size_mb / max(original_size_mb, 1e-6)) * 100.0

        if rectiq_result.history:
            final_row = rectiq_result.history[-1]
            logger.log(
                {
                    "rectiq_best_epoch": float(rectiq_result.best_epoch),
                    "rectiq_best_val_loss": float(rectiq_result.best_val_loss),
                    "rectiq_final_train_loss": float(final_row["train_loss"]),
                    "rectiq_final_val_loss": float(final_row["val_loss"]),
                    "rectiq_time_s": float(rectiq_time),
                }
            )
            if getattr(logger, "wandb_logger", None) is not None:
                logger.wandb_logger.log_summary(
                    {
                        "rectiq_best_epoch": int(rectiq_result.best_epoch),
                        "rectiq_best_val_loss": float(rectiq_result.best_val_loss),
                        "rectiq_final_train_loss": float(final_row["train_loss"]),
                        "rectiq_final_val_loss": float(final_row["val_loss"]),
                        "rectiq_time_s": float(rectiq_time),
                    }
                )

        precision_label = f"{base_precision_label}_RECTIQ"
        original_backbone = model.backbone
        model.backbone = quantized_backbone
        try:
            metrics = evaluate_model_on_detection_dataset(
                model=model,
                config=config,
                checkpoint_manager=checkpoint_manager,
                logger=logger,
                text_logger=text_logger,
                dataset_name=dataset_name,
                precision=precision_label,
            )
        finally:
            model.backbone = original_backbone

        q_stats = {
            "mode": "RECTIQ",
            "mode_description": f"Recti-Q on top of {base_precision_label}",
            "student_backend": student_backend,
            "original_size_mb": original_size_mb,
            "quantized_size_mb": effective_quantized_size_mb,
            "compression_ratio": compression_ratio,
            "size_reduction_pct": size_reduction_pct,
            "target_layers": "YOLO Detect pre-input LoRA adapter",
            "base_quantized_size_mb": base_quantized_size_mb,
            "adapter_size_mb": adapter_size_mb,
            "rectiq_rank": int(rectiq_cfg.rank),
            "rectiq_rank_per_scale": (
                [int(v) for v in rectiq_cfg.rank_per_scale]
                if getattr(rectiq_cfg, "rank_per_scale", None)
                else []
            ),
            "rectiq_alpha": float(rectiq_cfg.alpha),
            "rectiq_alpha_per_scale": (
                [float(v) for v in rectiq_cfg.alpha_per_scale]
                if getattr(rectiq_cfg, "alpha_per_scale", None)
                else []
            ),
            "rectiq_epochs": int(rectiq_cfg.epochs),
            "rectiq_best_epoch": int(rectiq_result.best_epoch),
            "rectiq_best_val_loss": float(rectiq_result.best_val_loss),
            "rectiq_time_s": float(rectiq_time),
            "rectiq_use_dwconv": bool(getattr(rectiq_cfg, "adapter_use_dwconv", True)),
            "rectiq_adapter_dropout": float(getattr(rectiq_cfg, "adapter_dropout", 0.0)),
            "rectiq_unfreeze_detect_cv3": bool(getattr(rectiq_cfg, "unfreeze_detect_cv3", False)),
            "rectiq_adapter_target": str(getattr(rectiq_cfg, "adapter_target", "detect_input")),
            "rectiq_train_split": train_split,
            "rectiq_val_split": val_split,
            "rectiq_train_batch_size": train_batch_size,
            "rectiq_val_batch_size": val_batch_size,
            "rectiq_adapter_path": str(adapter_path),
        }
        if export_path is not None:
            q_stats["export_path"] = str(export_path)
        if ptq_calibration_stats:
            q_stats["ptq_calibration_batches"] = int(ptq_calibration_stats.get("num_batches", 0))
            q_stats["ptq_num_features"] = int(ptq_calibration_stats.get("num_features", 0))
            q_stats["ptq_scale_mean"] = float(
                sum(ptq_calibration_stats.get("scales", [])) / max(len(ptq_calibration_stats.get("scales", [])), 1)
            )
            q_stats["ptq_use_ste"] = bool(rectiq_cfg.ptq_use_ste)
            q_stats["ptq_requantize_after_residual"] = bool(
                getattr(rectiq_cfg, "requantize_after_residual", True)
            )
        if student_closeness_stats:
            for k, v in student_closeness_stats.items():
                q_stats[f"student_closeness_{k}"] = float(v)

        if config.quantization.compute_detection_drift:
            fp32_view = SimpleNamespace(
                name=model.name,
                backbone=original_backbone,
                _yolo=getattr(model, "_yolo", None),
            )
            rectiq_view = SimpleNamespace(
                name=model.name,
                backbone=quantized_backbone,
                _yolo=getattr(model, "_yolo", None),
            )
            try:
                drift_metrics = evaluate_detection_quantization_drift(
                    fp32_model=fp32_view,
                    quantized_model=rectiq_view,
                    config=config,
                    logger=logger,
                    text_logger=text_logger,
                    precision_label=precision_label,
                    dataset_name=dataset_name,
                )
                for k, v in drift_metrics.items():
                    q_stats[f"drift_{k}"] = float(v)
            except Exception as e:
                text_logger.warning(
                    f"  [!] Decision drift failed for {model.name} [{precision_label}]: {e}"
                )

        return metrics, q_stats, precision_label
    except QuantizationSkipped:
        raise
    except Exception as e:
        raise QuantizationSkipped(
            f"Recti-Q pipeline failed for {model.name} [{base_precision_label}] on {dataset_name}: {e}"
        ) from e


def evaluate_quantized_classification(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
    mode: str,
) -> Tuple:
    """
    Quantize a classification model with torchao and evaluate on ImageNet.

    The model stays on GPU the entire time.

    Args:
        model: Original FP32 BaseModel.
        config: Experiment config.
        checkpoint_manager: For saving results.
        logger: Metrics logger.
        text_logger: Text logger.
        mode: One of QUANT_MODES keys ('W8A16', 'W8A8', 'W4A16', etc.).

    Returns:
        (ClassificationMetrics, quantization_stats_dict)
    """
    mode = resolve_mode(mode)  # handle aliases

    # ── Quantize the backbone ──
    text_logger.info(f"  Quantizing {model.name} – {mode}...")
    q_backbone, q_stats = quantize_model(
        model=model.backbone,
        mode=mode,
        device=config.device,
    )

    # Print quantization stats
    stats_fmt = format_quantization_stats(
        model_name=model.name,
        mode=mode,
        stats=q_stats,
    )
    print("\n" + stats_fmt)

    # ── Evaluate on ImageNet ──
    text_logger.info(f"  Evaluating {mode} on ImageNet (GPU)...")

    dataset_config = config.get_dataset("imagenet")
    dataloader = get_imagenet_loader(
        config=dataset_config,
        model_name=model.name,
        num_workers=config.num_workers,
        debug=config.debug,
        debug_samples=config.debug_samples,
    )

    # Swap backbone temporarily for inference
    original_backbone = model.backbone
    model.backbone = q_backbone

    try:
        results = run_inference(
            model=model,
            dataloader=dataloader,
            device=config.device,
            logger=logger,
            description=f"ImageNet - {model.name} [{mode}]",
        )
    finally:
        # Restore original backbone
        model.backbone = original_backbone

    metrics = results["metrics"]

    # Print results
    formatted = format_classification_results(
        model_name=model.name,
        dataset_name="ImageNet",
        metrics=metrics,
        precision=mode,
    )
    print("\n" + formatted)

    # Save
    checkpoint_manager.save_metrics(
        metrics=metrics.to_dict(),
        model_name=model.name,
        dataset_name="imagenet",
        precision=mode,
    )

    # Free quantized model
    del q_backbone
    _safe_cuda_empty_cache(text_logger=text_logger)

    return metrics, q_stats


def _is_yolo_model(model: BaseModel) -> bool:
    """Return True when model is our ultralytics-backed YOLO wrapper."""
    yolo_obj = getattr(model, "_yolo", None)
    return yolo_obj is not None and hasattr(yolo_obj, "model")


def _get_detection_quant_target(model: BaseModel) -> Tuple[str, torch.nn.Module]:
    """
    Return the torch module to quantize for non-YOLO detection models.

    YOLO uses an export-based quantization path, not torchao in-place quantization.
    """
    return "backbone", model.backbone


def _resolve_yolo_quant_mode(mode: str) -> str:
    """
    Resolve YOLO quant mode names.

    Supported:
      - YOLO_INT8 (canonical)
      - aliases: int8, yolo_int8, w8a8, w8a16, dynamic, weight_only
    """
    mode_key = mode.strip().lower()
    int8_aliases = {
        "yolo_int8",
        "int8",
        "w8a8",
        "w8a16",
        "dynamic",
        "weight_only",
        "weight_and_activation",
    }
    int4_aliases = {"int4", "w4a16", "w4a8fp"}
    if mode_key in int8_aliases:
        return "YOLO_INT8"
    if mode_key in int4_aliases:
        raise ValueError(
            f"YOLO mode '{mode}' is not supported. Ultralytics export path does not provide YOLO INT4."
        )
    if mode.strip().upper() == "YOLO_INT8":
        return "YOLO_INT8"
    raise ValueError(
        f"Unknown YOLO quantization mode '{mode}'. Use one of: YOLO_INT8 (or alias int8)."
    )


def _module_available(module_name: str) -> bool:
    """Return True if a Python module can be imported in this environment."""
    return importlib.util.find_spec(module_name) is not None


def _resolve_yolo_export_format(config: ExperimentConfig) -> str:
    """
    Resolve export format for YOLO INT8 and validate backend availability.

    Supported automatic policy:
      - CUDA device -> engine
      - CPU device  -> openvino
    """
    quant_cfg = config.quantization
    requested = (quant_cfg.yolo_format or "auto").lower()

    if requested == "auto":
        requested = "engine" if config.device.startswith("cuda") else "openvino"

    if requested == "openvino" and not _module_available("openvino"):
        raise QuantizationSkipped(
            "YOLO INT8 requires OpenVINO for yolo_format='openvino', "
            "but module 'openvino' is not installed in this environment. "
            "Install it (`pip install openvino`) or run on a TensorRT GPU node "
            "with yolo_format='engine'."
        )
    if requested == "openvino" and not _module_available("nncf"):
        raise QuantizationSkipped(
            "YOLO INT8 OpenVINO export also requires NNCF, but module 'nncf' is not installed. "
            "Install it (`pip install \"nncf>=2.14.0\"`) and rerun."
        )

    if requested == "engine":
        missing = []
        if not _module_available("onnx"):
            missing.append("onnx")
        if not _module_available("onnxslim"):
            missing.append("onnxslim")
        # Package is `onnxruntime-gpu`, import name is `onnxruntime`.
        if not _module_available("onnxruntime"):
            missing.append("onnxruntime-gpu")
        if not _module_available("tensorrt"):
            missing.append("tensorrt")
        if missing:
            raise QuantizationSkipped(
                "YOLO INT8 TensorRT export (yolo_format='engine') is missing required dependencies: "
                f"{', '.join(missing)}. "
                "Install them in your env, e.g. "
                "`pip install \"onnx>=1.12.0,<2.0.0\" onnxslim onnxruntime-gpu tensorrt`."
            )

    if requested == "engine" and not config.device.startswith("cuda"):
        raise QuantizationSkipped(
            "YOLO INT8 TensorRT export (yolo_format='engine') requires CUDA. "
            "Use a GPU node or set yolo_format='openvino' on CPU with OpenVINO installed."
        )

    return requested


def _get_path_size_mb(path: Path) -> float:
    """Compute total size in MB for a file or directory."""
    if path.is_file():
        return path.stat().st_size / (1024 ** 2)
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 ** 2)


def _count_images_from_yolo_split(split_value: Any) -> int:
    """
    Count image entries from a YOLO split value (path/list/txt/glob).
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if isinstance(split_value, (list, tuple)):
        return sum(_count_images_from_yolo_split(v) for v in split_value)

    split_path = Path(str(split_value))
    if split_path.is_file():
        if split_path.suffix.lower() == ".txt":
            with open(split_path, "r") as f:
                return sum(1 for line in f if line.strip())
        return 1 if split_path.suffix.lower() in image_exts else 0

    if split_path.is_dir():
        return sum(1 for p in split_path.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)

    split_str = str(split_value)
    if any(ch in split_str for ch in "*?[]"):
        parent = Path(".")
        pattern = split_str
        if "/" in split_str:
            parent = Path(split_str).parent
            pattern = Path(split_str).name
        if parent.exists():
            return sum(
                1
                for p in parent.glob(pattern)
                if p.is_file() and p.suffix.lower() in image_exts
            )

    return 0


def _count_yolo_calibration_images(data_arg: str, split: str = "val") -> Optional[int]:
    """
    Return total number of images available for YOLO INT8 calibration split.
    """
    try:
        from ultralytics.data.utils import check_det_dataset
    except Exception:
        return None

    try:
        data = check_det_dataset(data_arg)
        split_value = data.get(split) or data.get("val")
        if split_value is None:
            return None
        total = _count_images_from_yolo_split(split_value)
        return total if total > 0 else None
    except Exception:
        return None


def _resolve_yolo_calibration_fraction(
    quant_cfg,
    data_arg: str,
    text_logger,
) -> Tuple[float, str, Optional[int], Optional[int]]:
    """
    Resolve calibration amount for YOLO export.

    Precedence:
      1) quantization.num_calibration_batches
      2) quantization.calibration.num_samples
      3) quantization.yolo_fraction
    """
    base_fraction = float(quant_cfg.yolo_fraction)
    base_fraction = min(max(base_fraction, 1e-6), 1.0)

    total_images = _count_yolo_calibration_images(data_arg, split="val")
    num_batches = quant_cfg.num_calibration_batches
    num_samples = quant_cfg.calibration_num_samples

    if num_batches is not None and num_batches <= 0:
        text_logger.warning(
            f"  [!] Ignoring num_calibration_batches={num_batches}; expected positive integer."
        )
        num_batches = None
    if num_samples is not None and num_samples <= 0:
        text_logger.warning(
            f"  [!] Ignoring calibration.num_samples={num_samples}; expected positive integer."
        )
        num_samples = None

    if num_batches is not None and num_samples is not None:
        text_logger.info(
            "  Calibration precedence: using num_calibration_batches "
            "(calibration.num_samples ignored for this run)."
        )

    if num_batches is not None:
        target_images = int(num_batches) * int(quant_cfg.yolo_batch)
        if total_images is not None:
            eff_fraction = min(max(target_images / max(total_images, 1), 1e-6), 1.0)
        else:
            eff_fraction = base_fraction
            text_logger.warning(
                "  [!] Could not determine dataset size for num_calibration_batches; "
                f"falling back to yolo_fraction={base_fraction:.4f}."
            )
        return eff_fraction, "num_calibration_batches", target_images, total_images

    if num_samples is not None:
        target_images = int(num_samples)
        if total_images is not None:
            eff_fraction = min(max(target_images / max(total_images, 1), 1e-6), 1.0)
        else:
            eff_fraction = base_fraction
            text_logger.warning(
                "  [!] Could not determine dataset size for calibration.num_samples; "
                f"falling back to yolo_fraction={base_fraction:.4f}."
            )
        return eff_fraction, "calibration.num_samples", target_images, total_images

    target_images = int(round(base_fraction * total_images)) if total_images is not None else None
    return base_fraction, "yolo_fraction", target_images, total_images


def _normalize_cache_path(path_like: str) -> str:
    """Return a stable absolute path string when possible."""
    p = Path(str(path_like))
    try:
        if p.exists():
            return str(p.resolve())
    except Exception:
        pass
    return str(path_like)


def _infer_yolo_source_weights_id(model: BaseModel) -> str:
    """
    Infer a stable identifier for YOLO source weights.

    This is used in export cache keys so switching from COCO -> fine-tuned
    checkpoints cannot silently reuse stale TensorRT/OpenVINO exports.
    """
    yolo_obj = getattr(model, "_yolo", None)
    if yolo_obj is None:
        return "unknown"

    candidates = [
        getattr(yolo_obj, "ckpt_path", None),
        getattr(yolo_obj, "model_name", None),
        getattr(yolo_obj, "weights", None),
    ]
    for cand in candidates:
        if cand:
            return _normalize_cache_path(str(cand))

    return f"builtin:{getattr(model, 'name', 'yolo')}"


def _build_yolo_export_cache_key(
    config: ExperimentConfig,
    model_name: str,
    source_weights_id: str,
    canonical_mode: str,
    export_format: str,
    data_arg: str,
    quant_cfg,
    calib_fraction: float,
    calib_source: str,
) -> Dict[str, Any]:
    """
    Build a deterministic cache key for reusable YOLO exports.
    """
    key = {
        "key_version": 2,
        "seed": int(config.seed),
        "model_name": model_name,
        "source_weights": _normalize_cache_path(source_weights_id),
        "mode": canonical_mode,
        "format": export_format,
        "device": config.device,
        "data": _normalize_cache_path(data_arg),
        "imgsz": int(quant_cfg.yolo_imgsz),
        "batch": int(quant_cfg.yolo_batch),
        "fraction": round(float(calib_fraction), 8),
        "calibration_source": str(calib_source),
        "dynamic": bool(export_format == "engine"),
    }
    return key


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_json_file(path: Path, payload: Dict[str, Any]) -> None:
    """Save JSON payload atomically-ish for cache metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _yolo_export_manifest_path(export_dir: Path) -> Path:
    """Manifest path for reusable YOLO export metadata."""
    return export_dir / "qda_export_manifest.json"


def _relocate_yolo_export_artifacts(
    exported_path: Path,
    export_root: Path,
    export_name: str,
    text_logger,
) -> Path:
    """
    Move YOLO export artifacts (engine/onnx/cache) into a clean export subdirectory.

    Ultralytics TensorRT export writes files beside model weights, which may be the repo
    root. We relocate files with the same stem as exported_path for cleaner outputs.
    """
    if not exported_path.exists():
        return exported_path

    target_dir = (export_root / export_name).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    src_dir = exported_path.parent.resolve()
    if src_dir == target_dir:
        return exported_path

    stem = exported_path.stem
    movable_suffixes = {".engine", ".onnx", ".cache"}
    moved_files = []

    for file_path in src_dir.iterdir():
        if not file_path.is_file():
            continue
        if file_path.stem != stem:
            continue
        if file_path.suffix.lower() not in movable_suffixes:
            continue

        dest = target_dir / file_path.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(file_path), str(dest))
        moved_files.append(dest.name)

    relocated = target_dir / exported_path.name
    if relocated.exists():
        if moved_files:
            text_logger.info(
                f"  Moved YOLO export artifacts to {target_dir}: {', '.join(sorted(moved_files))}"
            )
        return relocated

    # Fallback: move exported file even if suffix filtering missed it.
    dest = target_dir / exported_path.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(exported_path), str(dest))
    moved_files.append(dest.name)
    text_logger.info(
        f"  Moved YOLO export artifacts to {target_dir}: {', '.join(sorted(moved_files))}"
    )
    return dest


def _strip_ultralytics_engine_header(engine_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    Split Ultralytics .engine files into raw TensorRT payload and metadata.

    Ultralytics prepends [4-byte metadata length][JSON metadata] before TRT bytes.
    """
    if len(engine_bytes) < 4:
        return engine_bytes, {}

    try:
        meta_len = int.from_bytes(engine_bytes[:4], byteorder="little")
        header_end = 4 + meta_len
        if meta_len <= 0 or header_end >= len(engine_bytes):
            return engine_bytes, {}
        metadata = json.loads(engine_bytes[4:header_end].decode("utf-8"))
        if isinstance(metadata, dict):
            return engine_bytes[header_end:], metadata
    except Exception:
        pass

    return engine_bytes, {}


def _layer_has_int8_tag(layer_info: Any) -> bool:
    """Return True when TensorRT layer inspector output contains any INT8 marker."""
    try:
        layer_text = json.dumps(layer_info, default=str).upper()
    except Exception:
        layer_text = str(layer_info).upper()
    return "INT8" in layer_text


def _collect_tensorrt_int8_coverage(engine_path: Path) -> Dict[str, Any]:
    """
    Inspect a TensorRT engine and estimate how many layers carry INT8 tags.
    """
    coverage: Dict[str, Any] = {
        "coverage_available": False,
        "coverage_error": None,
    }

    try:
        import tensorrt as trt
    except Exception as e:
        coverage["coverage_error"] = f"TensorRT import failed: {e}"
        return coverage

    try:
        raw_bytes = engine_path.read_bytes()
        payload, metadata = _strip_ultralytics_engine_header(raw_bytes)
        header_bytes = len(raw_bytes) - len(payload)

        logger = trt.Logger(trt.Logger.ERROR)
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(payload)
        if engine is None:
            coverage["coverage_error"] = "deserialize_cuda_engine() returned None"
            return coverage

        if not hasattr(trt, "LayerInformationFormat") or not hasattr(trt.LayerInformationFormat, "JSON"):
            coverage["coverage_error"] = "TensorRT JSON inspector format not available"
            return coverage

        inspector = engine.create_engine_inspector()
        info_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        parsed = json.loads(info_json)
        layers = parsed.get("Layers", []) if isinstance(parsed, dict) else []

        total_layers = len(layers)
        int8_layers = sum(1 for layer in layers if _layer_has_int8_tag(layer))
        fallback_layers = max(total_layers - int8_layers, 0)
        int8_ratio = (int8_layers / total_layers) if total_layers else 0.0
        fallback_ratio = (fallback_layers / total_layers) if total_layers else 0.0

        coverage.update(
            {
                "coverage_available": True,
                "coverage_error": None,
                "total_layers": total_layers,
                "int8_layers": int8_layers,
                "fallback_layers": fallback_layers,
                "int8_ratio": int8_ratio,
                "fallback_ratio": fallback_ratio,
                "engine_header_detected": bool(metadata),
                "engine_header_bytes": header_bytes,
            }
        )
    except Exception as e:
        coverage["coverage_error"] = str(e)

    return coverage


def _build_yolo_data_yaml(config: ExperimentConfig, dataset_name: str) -> Path:
    """
    Build a local YOLO dataset yaml for export INT8 calibration/evaluation.
    """
    if dataset_name == "bdd100k":
        return _build_bdd_data_yaml(config)

    if dataset_name != "coco":
        raise QuantizationSkipped(
            f"Unsupported detection dataset '{dataset_name}' for YOLO export YAML building."
        )

    import yaml

    dataset_cfg = config.get_dataset("coco")
    coco_root = Path(dataset_cfg.root).resolve()

    def _find_split_rel(split_name: str) -> Optional[str]:
        candidates = [split_name, f"images/{split_name}"]
        for rel in candidates:
            if (coco_root / rel).exists():
                return rel
        return None

    val_split = dataset_cfg.split or "val2017"
    val_rel = _find_split_rel(val_split)
    if val_rel is None:
        raise QuantizationSkipped(
            f"Could not find COCO split directory for '{val_split}' under '{coco_root}'. "
            f"Checked: '{coco_root / val_split}' and '{coco_root / ('images/' + val_split)}'."
        )

    train_rel = _find_split_rel("train2017")
    if train_rel is None:
        # Fall back to val split when train set is unavailable.
        train_rel = val_rel

    names = {i: f"class_{i}" for i in range(80)}
    data_yaml = {
        "path": str(coco_root),
        "train": train_rel,
        "val": val_rel,
        "names": names,
    }

    out_dir = Path(config.output.results_dir) / config.name / "yolo_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coco_yolo_export.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return out_path


def evaluate_quantized_yolo_detection(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
    mode: str,
    dataset_name: str,
    skip_eval: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Quantize YOLO via Ultralytics export (INT8) and evaluate on detection dataset.
    """
    from src.models.detection import YOLOWrapper

    canonical_mode = _resolve_yolo_quant_mode(mode)
    quant_cfg = config.quantization
    export_format = _resolve_yolo_export_format(config)
    valid_int8_formats = {
        "openvino",
        "engine",
        "coreml",
        "saved_model",
        "tflite",
        "tfjs",
        "mnn",
        "imx",
        "axelera",
    }
    if export_format not in valid_int8_formats:
        raise ValueError(
            f"YOLO export format '{export_format}' is not supported for int8. "
            f"Use one of: {sorted(valid_int8_formats)}"
        )
    precision_label = f"{canonical_mode}_{export_format}".upper()

    if not _is_yolo_model(model):
        raise ValueError("evaluate_quantized_yolo_detection() requires a YOLO model.")

    source_yolo = model._yolo
    source_nn = source_yolo.model
    original_size_mb = get_model_size_mb(source_nn)

    data_arg = quant_cfg.yolo_data or str(_build_yolo_data_yaml(config, dataset_name=dataset_name))
    calib_fraction, calib_source, calib_target_images, calib_total_images = _resolve_yolo_calibration_fraction(
        quant_cfg=quant_cfg,
        data_arg=data_arg,
        text_logger=text_logger,
    )

    export_root = Path(quant_cfg.yolo_export_dir or (Path(config.output.results_dir) / config.name / "yolo_exports"))
    export_root.mkdir(parents=True, exist_ok=True)
    export_name = f"{model.name}_{export_format}_int8"
    export_dir = (export_root / export_name).resolve()
    manifest_path = _yolo_export_manifest_path(export_dir)
    source_weights_id = _infer_yolo_source_weights_id(model)
    expected_cache_key = _build_yolo_export_cache_key(
        config=config,
        model_name=model.name,
        source_weights_id=source_weights_id,
        canonical_mode=canonical_mode,
        export_format=export_format,
        data_arg=data_arg,
        quant_cfg=quant_cfg,
        calib_fraction=calib_fraction,
        calib_source=calib_source,
    )

    text_logger.info(
        f"  Quantizing {model.name} via Ultralytics export – mode={canonical_mode}, format={export_format}"
    )

    # Optional ORT global thread-pool tuning.
    # Keep this OFF by default: enabling it can trigger "use_per_session_threads
    # must be false when using a global thread pool" warnings in some exporters.
    if os.environ.get("QDA_ORT_GLOBAL_THREAD_POOLS", "0") == "1":
        try:
            import onnxruntime as ort
            intra_threads = int(os.environ.get("QDA_ORT_INTRA_THREADS", "1"))
            inter_threads = int(os.environ.get("QDA_ORT_INTER_THREADS", "1"))
            ort.set_global_thread_pool_sizes(
                intra_op_num_threads=max(intra_threads, 1),
                inter_op_num_threads=max(inter_threads, 1),
            )
        except Exception:
            # Non-fatal: export can still proceed without this tuning.
            pass

    export_kwargs = {
        "format": export_format,
        "int8": True,
        "data": data_arg,
        "fraction": calib_fraction,
        "imgsz": quant_cfg.yolo_imgsz,
        "batch": quant_cfg.yolo_batch,
        "project": str(export_root),
        "name": export_name,
        "exist_ok": True,
        "verbose": False,
    }
    if calib_total_images is not None and calib_target_images is not None:
        actual_images = min(calib_target_images, calib_total_images)
        text_logger.info(
            "  INT8 calibration budget: "
            f"source={calib_source}, fraction={calib_fraction:.4f}, "
            f"images~{actual_images}/{calib_total_images}, batch={quant_cfg.yolo_batch}"
        )
    else:
        text_logger.info(
            "  INT8 calibration budget: "
            f"source={calib_source}, fraction={calib_fraction:.4f}, batch={quant_cfg.yolo_batch}"
        )
    if export_format == "engine":
        # TensorRT export needs explicit device on multi-device hosts.
        export_kwargs["device"] = config.device
        # Build dynamic TRT profiles so runtime can accept batch sizes that
        # differ from the calibration/export batch (e.g., batch 1 in wrapper).
        export_kwargs["dynamic"] = True

    exported_path: Optional[Path] = None
    export_reused = False

    if export_format == "engine" and quant_cfg.reuse_yolo_export:
        existing_export_dir = export_dir
        existing_candidates = sorted(
            existing_export_dir.glob("*.engine"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if existing_candidates and manifest_path.exists():
            manifest = _load_json_file(manifest_path)
            if manifest and manifest.get("cache_key") == expected_cache_key:
                exported_path = existing_candidates[0].resolve()
                export_reused = True
                text_logger.info(
                    f"  Reusing existing YOLO export: {exported_path} "
                    "(cache key matched; set quantization.reuse_yolo_export=false to force rebuild)"
                )
            else:
                text_logger.info(
                    "  Existing YOLO export cache key mismatch; rebuilding export "
                    "(e.g., seed/config/calibration changed)."
                )
        elif existing_candidates and not manifest_path.exists():
            text_logger.info(
                "  Found existing YOLO engine without cache manifest; rebuilding once to create "
                "seed/config-aware cache metadata."
            )

    if exported_path is None:
        t0 = time.time()
        try:
            exported = source_yolo.export(**export_kwargs)
        except Exception as e:
            raise QuantizationSkipped(
                f"YOLO INT8 export failed for format='{export_format}'. "
                f"Check backend dependencies and calibration data (quantization.yolo_data). "
                f"Original error: {e}"
            ) from e
        quant_time = time.time() - t0

        exported_path = Path(exported)
        if not exported_path.is_absolute():
            exported_path = (Path.cwd() / exported_path).resolve()
        exported_path = _relocate_yolo_export_artifacts(
            exported_path=exported_path,
            export_root=export_root,
            export_name=export_name,
            text_logger=text_logger,
        )
    else:
        quant_time = 0.0

    quantized_size_mb = _get_path_size_mb(exported_path)
    q_stats = {
        "mode": canonical_mode,
        "mode_description": f"YOLO export INT8 ({export_format})",
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": original_size_mb / max(quantized_size_mb, 1e-6),
        "size_reduction_pct": (1 - quantized_size_mb / max(original_size_mb, 1e-6)) * 100,
        "quantization_time_s": quant_time,
        "original_layers": {},
        "quantized_layers": {},
        "target_layers": f"Ultralytics export backend ({export_format})",
        "export_path": str(exported_path),
        "calibration_source": calib_source,
        "calibration_fraction": calib_fraction,
        "calibration_target_images": calib_target_images,
        "calibration_total_images": calib_total_images,
        "export_reused": export_reused,
    }
    if export_format == "engine" and not export_reused:
        _save_json_file(
            manifest_path,
            {
                "cache_key": expected_cache_key,
                "export_path": str(exported_path),
                "saved_at": datetime.now().isoformat(),
            },
        )

    if export_format == "engine":
        coverage = _collect_tensorrt_int8_coverage(exported_path)
        q_stats.update(coverage)

        if coverage.get("coverage_available"):
            total_layers = int(coverage["total_layers"])
            int8_layers = int(coverage["int8_layers"])
            fallback_layers = int(coverage["fallback_layers"])
            int8_ratio = float(coverage["int8_ratio"]) * 100.0
            fallback_ratio = float(coverage["fallback_ratio"]) * 100.0
            text_logger.info(
                "  TensorRT layer coverage: "
                f"INT8 {int8_layers}/{total_layers} ({int8_ratio:.1f}%), "
                f"fallback {fallback_layers}/{total_layers} ({fallback_ratio:.1f}%)"
            )
            if config.quantization.log_coverage_metrics:
                quant_log_metrics = {
                    "total_layers": total_layers,
                    "int8_layers": int8_layers,
                    "fallback_layers": fallback_layers,
                    "int8_ratio": float(coverage["int8_ratio"]),
                    "fallback_ratio": float(coverage["fallback_ratio"]),
                }
                logger.log(quant_log_metrics)
                if getattr(logger, "wandb_logger", None) is not None:
                    logger.wandb_logger.log_summary(
                        {
                            "total_layers": total_layers,
                            "int8_layers": int8_layers,
                            "fallback_layers": fallback_layers,
                            "int8_ratio": float(coverage["int8_ratio"]),
                            "fallback_ratio": float(coverage["fallback_ratio"]),
                        }
                    )
        else:
            text_logger.warning(
                "  [!] Could not inspect TensorRT layer coverage: "
                f"{coverage.get('coverage_error', 'unknown error')}"
            )

    stats_fmt = format_quantization_stats(
        model_name=model.name,
        mode=precision_label,
        stats=q_stats,
    )
    print("\n" + stats_fmt)

    # Evaluate using a fresh exported YOLO object so predictor/cache is aligned.
    from ultralytics import YOLO

    quantized_yolo = YOLO(str(exported_path), task="detect")
    quantized_yolo.overrides.update(
        {
            "task": "detect",
            "mode": "predict",
            "data": None,
            "verbose": False,
        }
    )
    quantized_backbone = YOLOWrapper(quantized_yolo)

    metrics: Dict[str, Any] = {}
    if skip_eval:
        text_logger.info(
            f"  Skipping base quantized evaluation for {model.name} [{precision_label}] "
            "(rectiq.skip_base_quant_eval=true)"
        )
    else:
        original_backbone = model.backbone
        model.backbone = quantized_backbone
        try:
            metrics = evaluate_model_on_detection_dataset(
                model=model,
                config=config,
                checkpoint_manager=checkpoint_manager,
                logger=logger,
                text_logger=text_logger,
                dataset_name=dataset_name,
                precision=precision_label,
            )
        finally:
            model.backbone = original_backbone

    if (not skip_eval) and config.quantization.compute_detection_drift:
        fp32_view = SimpleNamespace(
            name=model.name,
            backbone=model.backbone,
            _yolo=getattr(model, "_yolo", None),
        )
        quant_view = SimpleNamespace(
            name=model.name,
            backbone=quantized_backbone,
            _yolo=getattr(model, "_yolo", None),
        )
        try:
            drift_metrics = evaluate_detection_quantization_drift(
                fp32_model=fp32_view,
                quantized_model=quant_view,
                config=config,
                logger=logger,
                text_logger=text_logger,
                precision_label=precision_label,
                dataset_name=dataset_name,
            )
            for k, v in drift_metrics.items():
                q_stats[f"drift_{k}"] = float(v)
        except Exception as e:
            text_logger.warning(
                f"  [!] Decision drift failed for {model.name} [{precision_label}]: {e}"
            )
    else:
        text_logger.info(
            f"  Skipping decision drift metrics for {model.name} [{precision_label}] "
            "(quantization.compute_detection_drift=false)"
        )

    return metrics, q_stats, precision_label


def evaluate_quantized_detection(
    model: BaseModel,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    logger: MetricsLogger,
    text_logger,
    mode: str,
    dataset_name: str,
    skip_eval: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Quantize a detection model and evaluate on one detection dataset.

    - YOLO: Ultralytics export INT8 backend path.
    - Torchvision detection: torchao path.
    """
    if _is_yolo_model(model):
        return evaluate_quantized_yolo_detection(
            model=model,
            config=config,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            text_logger=text_logger,
            mode=mode,
            dataset_name=dataset_name,
            skip_eval=skip_eval,
        )

    mode = resolve_mode(mode)
    _, target_module = _get_detection_quant_target(model)

    text_logger.info(f"  Quantizing {model.name} – {mode} (backbone)...")
    q_module, q_stats = quantize_model(
        model=target_module,
        mode=mode,
        device=config.device,
    )

    stats_fmt = format_quantization_stats(
        model_name=model.name,
        mode=mode,
        stats=q_stats,
    )
    print("\n" + stats_fmt)

    # Swap in quantized module for evaluation
    original_module = model.backbone
    model.backbone = q_module

    if skip_eval:
        metrics = {}
        model.backbone = original_module
    else:
        try:
            metrics = evaluate_model_on_detection_dataset(
                model=model,
                config=config,
                checkpoint_manager=checkpoint_manager,
                logger=logger,
                text_logger=text_logger,
                dataset_name=dataset_name,
                precision=mode,
            )
        finally:
            model.backbone = original_module

    if (not skip_eval) and config.quantization.compute_detection_drift:
        fp32_view = SimpleNamespace(name=model.name, backbone=original_module)
        quant_view = SimpleNamespace(name=model.name, backbone=q_module)
        try:
            drift_metrics = evaluate_detection_quantization_drift(
                fp32_model=fp32_view,
                quantized_model=quant_view,
                config=config,
                logger=logger,
                text_logger=text_logger,
                precision_label=mode,
                dataset_name=dataset_name,
            )
            for k, v in drift_metrics.items():
                q_stats[f"drift_{k}"] = float(v)
        except Exception as e:
            text_logger.warning(
                f"  [!] Decision drift failed for {model.name} [{mode}]: {e}"
            )
    else:
        text_logger.info(
            f"  Skipping decision drift metrics for {model.name} [{mode}] "
            "(quantization.compute_detection_drift=false)"
        )

    # Free quantized model
    del q_module
    _safe_cuda_empty_cache(text_logger=text_logger)

    return metrics, q_stats, mode


def _log_wandb_comparison(
    wandb_logger,
    model_name: str,
    all_results: Dict[str, Any],
):
    """
    Log a wandb comparison table + bar charts for one model.

    Creates a single table + bar charts for either:
      - classification: [precision, top1, top5, size_mb, compression_ratio]
      - detection: [precision, mAP, mAP50, size_mb, compression_ratio]
    """
    if wandb_logger is None or not wandb_logger.enabled:
        return

    try:
        import wandb
    except ImportError:
        return

    # Detect task type from first available metrics entry
    first_metrics = None
    for entry in all_results.values():
        if entry.get("metrics") is not None:
            first_metrics = entry["metrics"]
            break

    if first_metrics is None:
        return

    is_detection = isinstance(first_metrics, dict) and "mAP" in first_metrics

    if is_detection:
        columns = ["precision", "mAP", "mAP50", "size_mb", "compression_ratio"]
        rows = []
        for prec_key, entry in all_results.items():
            metrics = entry.get("metrics")
            stats = entry.get("stats")
            if not isinstance(metrics, dict):
                continue
            size = stats.get("quantized_size_mb", stats.get("original_size_mb", 0)) if stats else 0
            cr = stats.get("compression_ratio", 1.0) if stats else 1.0
            rows.append([prec_key, metrics.get("mAP", 0), metrics.get("mAP50", 0), size, cr])
        if not rows:
            return

        table = wandb.Table(columns=columns, data=rows)
        wandb.log({f"{model_name}/comparison_table": table})
        wandb.log({
            f"{model_name}/map_comparison": wandb.plot.bar(
                table, "precision", "mAP",
                title=f"{model_name} – COCO mAP by Precision",
            ),
            f"{model_name}/map50_comparison": wandb.plot.bar(
                table, "precision", "mAP50",
                title=f"{model_name} – COCO mAP50 by Precision",
            ),
            f"{model_name}/size_comparison": wandb.plot.bar(
                table, "precision", "size_mb",
                title=f"{model_name} – Model Size (MB) by Precision",
            ),
            f"{model_name}/compression_comparison": wandb.plot.bar(
                table, "precision", "compression_ratio",
                title=f"{model_name} – Compression Ratio by Precision",
            ),
        })
        return

    # Classification table/charts
    columns = ["precision", "top1", "top5", "size_mb", "compression_ratio"]
    rows = []
    for prec_key, entry in all_results.items():
        metrics = entry.get("metrics")
        stats = entry.get("stats")
        if metrics is None:
            continue
        t1 = metrics.top1_accuracy if hasattr(metrics, "top1_accuracy") else metrics.get("top1_accuracy", 0)
        t5 = metrics.top5_accuracy if hasattr(metrics, "top5_accuracy") else metrics.get("top5_accuracy", 0)
        size = stats.get("quantized_size_mb", stats.get("original_size_mb", 0)) if stats else 0
        cr = stats.get("compression_ratio", 1.0) if stats else 1.0
        rows.append([prec_key, t1, t5, size, cr])
    if not rows:
        return

    table = wandb.Table(columns=columns, data=rows)
    wandb.log({f"{model_name}/comparison_table": table})
    wandb.log({
        f"{model_name}/top1_comparison": wandb.plot.bar(
            table, "precision", "top1",
            title=f"{model_name} – Top-1 Accuracy by Precision",
        ),
        f"{model_name}/top5_comparison": wandb.plot.bar(
            table, "precision", "top5",
            title=f"{model_name} – Top-5 Accuracy by Precision",
        ),
        f"{model_name}/size_comparison": wandb.plot.bar(
            table, "precision", "size_mb",
            title=f"{model_name} – Model Size (MB) by Precision",
        ),
        f"{model_name}/compression_comparison": wandb.plot.bar(
            table, "precision", "compression_ratio",
            title=f"{model_name} – Compression Ratio by Precision",
        ),
    })


def _slugify_wandb_token(value: str) -> str:
    """Convert run name tokens to a readable, wandb-friendly form."""
    allowed = []
    for ch in value.lower():
        if ch.isalnum():
            allowed.append(ch)
        else:
            allowed.append("-")
    token = "".join(allowed)
    while "--" in token:
        token = token.replace("--", "-")
    return token.strip("-")


def _build_wandb_phase_name(
    model_name: str,
    task: str,
    precision: str,
    dataset_name: Optional[str] = None,
) -> str:
    """Build consistent run names like 'yolov8n-detection-bdd100k-fp32'."""
    tokens = [
        _slugify_wandb_token(model_name),
        _slugify_wandb_token(task),
    ]
    if dataset_name:
        tokens.append(_slugify_wandb_token(dataset_name))
    tokens.append(_slugify_wandb_token(precision))
    return "-".join(tokens)


def _build_wandb_phase_loggers(
    config: ExperimentConfig,
    model_name: str,
    task: str,
    precision: str,
    dataset_name: Optional[str] = None,
) -> Tuple[Optional[WandbLogger], MetricsLogger]:
    """
    Create per-phase wandb + metrics logger.

    Each (model, precision) phase is logged as its own wandb run to simplify
    run-level comparisons in the UI sidebar.
    """
    if not config.logging.wandb.enabled:
        return None, MetricsLogger(config, None)

    run_name = _build_wandb_phase_name(
        model_name=model_name,
        task=task,
        precision=precision,
        dataset_name=dataset_name,
    )
    run_group = _slugify_wandb_token(config.name)
    base_tags = [
        f"model:{model_name}",
        f"task:{task}",
        f"precision:{precision}",
    ]
    if dataset_name:
        base_tags.append(f"dataset:{dataset_name}")
    run_config = {
        "phase_model": model_name,
        "phase_task": task,
        "phase_precision": precision,
    }
    if dataset_name:
        run_config["phase_dataset"] = dataset_name
    job_tokens = [_slugify_wandb_token(task)]
    if dataset_name:
        job_tokens.append(_slugify_wandb_token(dataset_name))
    job_tokens.append(_slugify_wandb_token(precision))
    raw_job_type = "_".join(job_tokens)
    if len(raw_job_type) > 64:
        # WandB enforces a 64-char job_type limit.
        import hashlib

        suffix = hashlib.sha1(raw_job_type.encode("utf-8")).hexdigest()[:8]
        head_len = 64 - 1 - len(suffix)
        raw_job_type = f"{raw_job_type[:head_len]}_{suffix}"
    wandb_logger = WandbLogger(
        config=config,
        run_name=run_name,
        run_group=run_group,
        run_job_type=raw_job_type,
        extra_tags=base_tags,
        extra_config=run_config,
    )
    return wandb_logger, MetricsLogger(config, wandb_logger)


def _is_summary_metric_payload(metrics: Any) -> bool:
    """
    Return True if value looks like an evaluation metrics payload.
    """
    if hasattr(metrics, "top1_accuracy") and hasattr(metrics, "top5_accuracy"):
        return True
    if isinstance(metrics, dict):
        metric_keys = {"mAP", "mAP50", "top1_accuracy", "top5_accuracy"}
        return any(k in metrics for k in metric_keys)
    return False


def _select_summary_metrics_payload(precision_results: Any) -> Any:
    """
    Select one metrics payload for end-of-run summary.

    Priority:
      1) fp32 baseline (when present)
      2) Recti-Q result
      3) any other valid precision payload
    """
    if not isinstance(precision_results, dict):
        return precision_results

    fp32_metrics = precision_results.get("fp32")
    if _is_summary_metric_payload(fp32_metrics):
        return fp32_metrics

    rectiq_candidates = []
    other_candidates = []
    for key, value in precision_results.items():
        if not _is_summary_metric_payload(value):
            continue
        key_str = str(key).lower()
        if "rectiq" in key_str:
            rectiq_candidates.append((key_str, value))
        else:
            other_candidates.append((key_str, value))

    if rectiq_candidates:
        rectiq_candidates.sort(key=lambda x: x[0])
        return rectiq_candidates[0][1]
    if other_candidates:
        other_candidates.sort(key=lambda x: x[0])
        return other_candidates[0][1]

    return {}


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.debug:
        config.debug = True
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    if args.no_wandb:
        config.logging.wandb.enabled = False
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup logging
    text_logger = setup_logging(config)
    text_logger.info(f"Starting experiment: {config.name}")
    text_logger.info(f"Configuration: {args.config}")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        text_logger.warning(
            f"CUDA requested ({config.device}) but not available. Falling back to CPU."
        )
        config.device = "cpu"
    text_logger.info(f"Device: {config.device}")
    text_logger.info(f"Debug mode: {config.debug}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(config)
    
    # Determine which datasets to evaluate
    datasets_to_eval = args.datasets or list(config.datasets.keys())
    
    # Determine which models to evaluate
    models_to_eval = args.models
    if models_to_eval:
        model_configs = [config.get_model(name) for name in models_to_eval]
    else:
        model_configs = config.models
    
    # Print experiment header
    header = format_experiment_header(
        experiment_name=config.name,
        config_file=args.config,
        device=config.device,
        models=[m.name for m in model_configs],
        datasets=datasets_to_eval,
    )
    text_logger.info(header)
    
    # Track task types for final summary
    task_types = {}
    for ds_name in datasets_to_eval:
        if ds_name in {"coco", "bdd100k"}:
            task_types[ds_name] = "detection"
        else:
            task_types[ds_name] = "classification"
    
    # Results storage
    all_results = {}
    
    # Determine quantization modes to run
    quant_enabled = config.quantization.enabled
    quant_modes = config.quantization.modes if quant_enabled else []
    
    if quant_enabled:
        text_logger.info(f"  Quantization ENABLED – modes: {quant_modes}")
    else:
        text_logger.info(f"  Quantization DISABLED – running FP32 baselines only")
    
    # Evaluate each model
    for model_config in model_configs:
        # Create model
        model = ModelFactory.create(model_config, device=config.device)
        model_info = model.get_model_info()
        
        # Print model header
        model_header = format_model_header(model_config.name, model_info)
        text_logger.info(model_header)
        
        all_results[model_config.name] = {}
        
        # ── Phase 1: FP32 Baseline ──
        fp32_results = {}
        if model.task == "detection":
            candidate_datasets = [ds for ds in datasets_to_eval if ds in {"coco", "bdd100k"}]
        else:
            candidate_datasets = [ds for ds in datasets_to_eval if ds not in {"coco", "bdd100k"}]
        phase_dataset = candidate_datasets[0] if candidate_datasets else None

        rectiq_only_mode = (
            model.task == "detection"
            and _is_yolo_model(model)
            and config.rectiq.enabled
            and config.rectiq.only
        )
        skip_fp32_phase_mode = (
            model.task == "detection"
            and _is_yolo_model(model)
            and config.quantization.skip_fp32_phase
        )
        fp32_wandb_logger: Optional[WandbLogger] = None
        fp32_metrics_logger = MetricsLogger(config, None)

        if rectiq_only_mode or skip_fp32_phase_mode:
            reason = (
                "Recti-Q only mode enabled"
                if rectiq_only_mode
                else "quantization.skip_fp32_phase=true"
            )
            text_logger.info(
                f"  {reason}: skipping FP32 phase wandb run for this model."
            )
        else:
            fp32_wandb_logger, fp32_metrics_logger = _build_wandb_phase_loggers(
                config=config,
                model_name=model_config.name,
                task=model.task,
                precision="fp32",
                dataset_name=phase_dataset,
            )

            for dataset_name in datasets_to_eval:
                if dataset_name not in config.datasets:
                    text_logger.warning(f"  [!] Dataset {dataset_name} not in config, skipping")
                    continue

                try:
                    if dataset_name == "imagenet":
                        metrics = evaluate_model_on_imagenet(
                            model=model,
                            config=config,
                            checkpoint_manager=checkpoint_manager,
                            logger=fp32_metrics_logger,
                            text_logger=text_logger,
                        )
                        fp32_results[dataset_name] = metrics
                        all_results[model_config.name][dataset_name] = {"fp32": metrics}

                    elif dataset_name == "imagenet_c":
                        metrics = evaluate_model_on_imagenet_c(
                            model=model,
                            config=config,
                            checkpoint_manager=checkpoint_manager,
                            logger=fp32_metrics_logger,
                            text_logger=text_logger,
                        )
                        fp32_results[dataset_name] = metrics
                        all_results[model_config.name][dataset_name] = {"fp32": metrics}

                    elif dataset_name in {"coco", "bdd100k"}:
                        metrics = evaluate_model_on_detection_dataset(
                            model=model,
                            config=config,
                            checkpoint_manager=checkpoint_manager,
                            logger=fp32_metrics_logger,
                            text_logger=text_logger,
                            dataset_name=dataset_name,
                        )
                        fp32_results[dataset_name] = metrics
                        all_results[model_config.name][dataset_name] = {"fp32": metrics}

                    else:
                        text_logger.warning(
                            f"Dataset {dataset_name} not yet supported, skipping"
                        )

                except FileNotFoundError as e:
                    text_logger.error(f"  [!] Dataset not found: {e}")
                    continue
                except QuantizationSkipped as e:
                    text_logger.warning(
                        f"  [!] Skipping baseline for {model_config.name} on {dataset_name}: {e}"
                    )
                    continue
                except Exception as e:
                    text_logger.error(f"  [!] Error evaluating on {dataset_name}: {e}")
                    raise
        # ── Phase 2: Quantized Evaluation ──
        # Collect per-precision results for wandb comparison chart
        model_comparison = {}  # precision_key -> {"metrics": ..., "stats": ...}

        # Add FP32 baseline to comparison (task-specific reference dataset)
        fp32_reference_metrics = None
        reference_dataset = None
        reference_task = model.task

        if model.task == "classification":
            fp32_reference_metrics = fp32_results.get("imagenet")
            reference_dataset = "imagenet"
            quant_target = model.backbone
        elif model.task == "detection":
            detection_datasets = [ds for ds in datasets_to_eval if ds in {"coco", "bdd100k"}]
            reference_dataset = detection_datasets[0] if detection_datasets else None
            fp32_reference_metrics = fp32_results.get(reference_dataset) if reference_dataset else None
            if _is_yolo_model(model):
                quant_target = model._yolo.model
            else:
                _, quant_target = _get_detection_quant_target(model)
        else:
            quant_target = model.backbone

        if not (rectiq_only_mode or skip_fp32_phase_mode):
            fp32_size = get_model_size_mb(quant_target)
            if config.logging.wandb.log_model_size:
                fp32_size_metrics = {"model_size_mb": float(fp32_size)}
                fp32_metrics_logger.log(fp32_size_metrics)
                if getattr(fp32_metrics_logger, "wandb_logger", None) is not None:
                    fp32_metrics_logger.wandb_logger.log_summary(
                        {
                            "model_size_mb": fp32_size,
                        }
                    )

            if fp32_reference_metrics is not None:
                model_comparison["FP32"] = {
                    "metrics": fp32_reference_metrics,
                    "stats": {
                        "original_size_mb": fp32_size,
                        "quantized_size_mb": fp32_size,
                        "compression_ratio": 1.0,
                        "size_reduction_pct": 0.0,
                    },
                }
            if fp32_wandb_logger:
                fp32_wandb_logger.finish()

        if quant_enabled and quant_modes:
            for qmode in quant_modes:
                run_mode = qmode
                try:
                    if model.task == "classification":
                        run_mode = resolve_mode(qmode)
                    elif model.task == "detection" and not _is_yolo_model(model):
                        run_mode = resolve_mode(qmode)
                except ValueError as e:
                    text_logger.error(f"  [!] Invalid quantization mode '{qmode}': {e}")
                    continue

                text_logger.info(f"\n  ── {run_mode} quantization ──")
                if model.task == "detection" and _is_yolo_model(model):
                    try:
                        yolo_mode = _resolve_yolo_quant_mode(run_mode)
                        yolo_format = _resolve_yolo_export_format(config)
                        phase_precision = f"{yolo_mode}_{yolo_format}".lower()
                        if not config.rectiq.enabled:
                            phase_precision = f"{phase_precision}_rectiq_disabled"
                    except Exception:
                        phase_precision = str(run_mode).lower()
                else:
                    phase_precision = str(run_mode).lower()
                log_base_quant_run = not (
                    rectiq_only_mode and model.task == "detection" and _is_yolo_model(model)
                )
                if log_base_quant_run:
                    q_wandb_logger, q_metrics_logger = _build_wandb_phase_loggers(
                        config=config,
                        model_name=model_config.name,
                        task=model.task,
                        precision=phase_precision,
                        dataset_name=phase_dataset,
                    )
                else:
                    q_wandb_logger = None
                    q_metrics_logger = MetricsLogger(config, None)

                try:
                    base_quant_metrics_available = True
                    if model.task == "classification":
                        q_metrics, q_stats = evaluate_quantized_classification(
                            model=model,
                            config=config,
                            checkpoint_manager=checkpoint_manager,
                            logger=q_metrics_logger,
                            text_logger=text_logger,
                            mode=run_mode,
                        )
                        result_key = run_mode

                        # Store results
                        if "imagenet" in all_results[model_config.name]:
                            all_results[model_config.name]["imagenet"][result_key] = q_metrics
                        else:
                            all_results[model_config.name]["imagenet"] = {result_key: q_metrics}

                    elif model.task == "detection":
                        detection_datasets = [ds for ds in datasets_to_eval if ds in {"coco", "bdd100k"}]
                        if not detection_datasets:
                            text_logger.info(
                                "  Skipping detection quantization: no detection dataset selected "
                                "(supported: coco, bdd100k)"
                            )
                            continue
                        if len(detection_datasets) > 1:
                            text_logger.info(
                                "  Multiple detection datasets selected; "
                                f"using '{detection_datasets[0]}' for quantized evaluation."
                            )
                        detection_dataset = detection_datasets[0]
                        skip_base_quant_eval = (
                            config.rectiq.enabled
                            and config.rectiq.skip_base_quant_eval
                            and _is_yolo_model(model)
                        )

                        q_metrics, q_stats, result_key = evaluate_quantized_detection(
                            model=model,
                            config=config,
                            checkpoint_manager=checkpoint_manager,
                            logger=q_metrics_logger,
                            text_logger=text_logger,
                            mode=run_mode,
                            dataset_name=detection_dataset,
                            skip_eval=skip_base_quant_eval,
                        )

                        base_quant_metrics_available = not (
                            isinstance(q_metrics, dict) and len(q_metrics) == 0
                        )
                        if base_quant_metrics_available:
                            if detection_dataset in all_results[model_config.name]:
                                all_results[model_config.name][detection_dataset][result_key] = q_metrics
                            else:
                                all_results[model_config.name][detection_dataset] = {result_key: q_metrics}
                        else:
                            text_logger.info(
                                f"  Base quantized evaluation skipped for {model_config.name} [{result_key}] "
                                "(rectiq.skip_base_quant_eval=true)."
                            )

                    else:
                        text_logger.info(
                            f"  Skipping quantization for {model_config.name} "
                            f"(unsupported task={model.task})"
                        )
                        continue

                    # Store for comparison chart
                    if base_quant_metrics_available:
                        model_comparison[result_key] = {
                            "metrics": q_metrics,
                            "stats": q_stats,
                        }

                    if (
                        config.logging.wandb.log_model_size
                        and getattr(q_metrics_logger, "wandb_logger", None) is not None
                    ):
                        q_size = q_stats.get(
                            "quantized_size_mb",
                            q_stats.get("original_size_mb", 0.0),
                        )
                        q_size_metrics = {"model_size_mb": float(q_size)}
                        q_metrics_logger.log(q_size_metrics)
                        q_metrics_logger.wandb_logger.log_summary(
                            {
                                "model_size_mb": q_size,
                            }
                        )

                    # Print comparison vs FP32
                    if (
                        base_quant_metrics_available
                        and fp32_reference_metrics is not None
                        and reference_dataset is not None
                    ):
                        comparison = format_comparison_row(
                            model_name=model_config.name,
                            dataset_name=reference_dataset,
                            task=reference_task,
                            fp32_metrics=fp32_reference_metrics,
                            quant_metrics=q_metrics,
                            quant_mode=result_key,
                            quant_stats=q_stats,
                        )
                        print("\n" + comparison)

                    if (
                        model.task == "detection"
                        and _is_yolo_model(model)
                        and config.rectiq.enabled
                    ):
                        teacher_suffix = (
                            "with_teacher"
                            if config.rectiq.use_teacher and float(config.rectiq.feature_kd_weight) > 0.0
                            else "without_teacher"
                        )
                        rectiq_phase_precision = (
                            f"{result_key}_rectiq_enabled_{teacher_suffix}"
                        ).lower()
                        r_wandb_logger, r_metrics_logger = _build_wandb_phase_loggers(
                            config=config,
                            model_name=model_config.name,
                            task=model.task,
                            precision=rectiq_phase_precision,
                            dataset_name=detection_dataset,
                        )
                        try:
                            r_metrics, r_stats, r_result_key = evaluate_rectiq_yolo_detection(
                                model=model,
                                config=config,
                                checkpoint_manager=checkpoint_manager,
                                logger=r_metrics_logger,
                                text_logger=text_logger,
                                dataset_name=detection_dataset,
                                base_precision_label=result_key,
                                quant_stats=q_stats,
                            )

                            if detection_dataset in all_results[model_config.name]:
                                all_results[model_config.name][detection_dataset][r_result_key] = r_metrics
                            else:
                                all_results[model_config.name][detection_dataset] = {r_result_key: r_metrics}

                            model_comparison[r_result_key] = {
                                "metrics": r_metrics,
                                "stats": r_stats,
                            }

                            if (
                                config.logging.wandb.log_model_size
                                and getattr(r_metrics_logger, "wandb_logger", None) is not None
                            ):
                                r_size = r_stats.get(
                                    "quantized_size_mb",
                                    r_stats.get("original_size_mb", 0.0),
                                )
                                r_metrics_logger.log({"model_size_mb": float(r_size)})
                                r_metrics_logger.wandb_logger.log_summary(
                                    {"model_size_mb": float(r_size)}
                                )

                            if (
                                config.quantization.log_coverage_metrics
                                and getattr(r_metrics_logger, "wandb_logger", None) is not None
                            ):
                                # Recti-Q runs may skip base quant phase logging, so forward
                                # INT8 runtime coverage from base quant stats into Recti-Q run.
                                coverage_stats = r_stats
                                if "total_layers" not in coverage_stats:
                                    coverage_stats = q_stats
                                coverage_keys = (
                                    "total_layers",
                                    "int8_layers",
                                    "fallback_layers",
                                    "int8_ratio",
                                    "fallback_ratio",
                                )
                                coverage_payload = {
                                    k: float(coverage_stats[k])
                                    for k in coverage_keys
                                    if k in coverage_stats
                                }
                                if coverage_payload:
                                    r_metrics_logger.log(coverage_payload)
                                    r_metrics_logger.wandb_logger.log_summary(
                                        coverage_payload
                                    )

                            if fp32_reference_metrics is not None and reference_dataset is not None:
                                rectiq_cmp = format_comparison_row(
                                    model_name=model_config.name,
                                    dataset_name=reference_dataset,
                                    task=reference_task,
                                    fp32_metrics=fp32_reference_metrics,
                                    quant_metrics=r_metrics,
                                    quant_mode=r_result_key,
                                    quant_stats=r_stats,
                                )
                                print("\n" + rectiq_cmp)
                        except QuantizationSkipped as e:
                            if rectiq_only_mode:
                                raise RuntimeError(
                                    "Recti-Q only mode requested, but Recti-Q could not run: "
                                    f"{e}"
                                ) from e
                            text_logger.warning(
                                f"  [!] Skipping Recti-Q for {model_config.name} [{result_key}]: {e}"
                            )
                        except Exception as e:
                            if rectiq_only_mode:
                                raise RuntimeError(
                                    "Recti-Q only mode requested, but Recti-Q crashed: "
                                    f"{e}"
                                ) from e
                            text_logger.error(
                                f"  [!] Error during Recti-Q for {model_config.name} [{result_key}]: {e}"
                            )
                        finally:
                            if r_wandb_logger:
                                r_wandb_logger.finish()

                except QuantizationSkipped as e:
                    text_logger.warning(
                        f"  [!] Skipping {run_mode} for {model_config.name}: {e}"
                    )
                    continue
                except Exception as e:
                    text_logger.error(
                        f"  [!] Error during {run_mode} for {model_config.name}: {e}"
                    )
                    import traceback
                    traceback.print_exc()
                    continue
                finally:
                    if q_wandb_logger:
                        q_wandb_logger.finish()

        # Comparison charts are omitted when logging each precision as a separate
        # wandb run. Run-level comparison is then handled directly in wandb UI.
        
        # Clear model from memory
        del model
        _safe_cuda_empty_cache(text_logger=text_logger)
    
    # Log final summary using unified formatting
    # Flatten nested results for backward-compat with format_final_summary
    flat_results = {}
    for mname, ds_dict in all_results.items():
        flat_results[mname] = {}
        for ds_name, prec_dict in ds_dict.items():
            flat_results[mname][ds_name] = _select_summary_metrics_payload(prec_dict)
    
    summary = format_final_summary(flat_results, task_types)
    text_logger.info(summary)
    
    text_logger.info("Experiment completed successfully!")
    
    return all_results


if __name__ == "__main__":
    main()
