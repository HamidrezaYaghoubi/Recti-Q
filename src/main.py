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
import sys
from datetime import datetime
from pathlib import Path
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
from src.datasets import get_imagenet_loader, get_imagenet_c_loader, get_all_imagenet_c_loaders
from src.evaluation import MetricsComputer, ClassificationMetrics
from src.quantization import quantize_model, QUANT_MODES


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
) -> Dict[str, Any]:
    """
    Evaluate a detection model on COCO dataset with proper mAP computation.
    
    Args:
        model: Model to evaluate.
        config: Experiment configuration.
        checkpoint_manager: Checkpoint manager.
        logger: Metrics logger.
        text_logger: Text logger.
        
    Returns:
        Dictionary with detection metrics (mAP, mAP50, mAP75, etc.).
    """
    import sys
    import io
    from src.datasets import get_coco_loader
    from src.evaluation import COCOEvaluator, remap_yolo_to_coco_labels
    from pycocotools.coco import COCO
    
    text_logger.info(f"  Evaluating on COCO val2017...")
    
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
        precision="fp32",
        verbose=True,  # Show size breakdown
    )
    print("\n" + formatted)
    
    # Log to wandb
    logger.log({
        f"{model.name}/coco/mAP": metrics["mAP"],
        f"{model.name}/coco/mAP50": metrics["mAP50"],
        f"{model.name}/coco/mAP75": metrics["mAP75"],
        f"{model.name}/coco/mAP_small": metrics["mAP_small"],
        f"{model.name}/coco/mAP_medium": metrics["mAP_medium"],
        f"{model.name}/coco/mAP_large": metrics["mAP_large"],
        f"{model.name}/coco/AR_100": metrics["AR_100"],
        f"{model.name}/coco/num_images": metrics["num_images"],
        f"{model.name}/coco/total_detections": metrics["total_detections"],
    })
    
    # Save predictions
    if config.output.save_predictions:
        checkpoint_manager.save_predictions(
            predictions={"predictions": all_predictions, "targets": all_targets, "metrics": metrics},
            model_name=model.name,
            dataset_name="coco",
            precision="fp32",
        )
    
    # Save metrics
    checkpoint_manager.save_metrics(
        metrics=metrics,
        model_name=model.name,
        dataset_name="coco",
        precision="fp32",
    )
    
    return metrics


# ========================================================================
# Quantized evaluation helpers
# ========================================================================

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
        mode: 'weight_only' or 'weight_and_activation'.

    Returns:
        (ClassificationMetrics, quantization_stats_dict)
    """
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
    text_logger.info(f"  Evaluating INT8 {mode} on ImageNet (GPU)...")

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
            description=f"ImageNet - {model.name} [INT8 {mode}]",
        )
    finally:
        # Restore original backbone
        model.backbone = original_backbone

    metrics = results["metrics"]
    precision_tag = f"int8_{mode}"

    # Print results
    formatted = format_classification_results(
        model_name=model.name,
        dataset_name="ImageNet",
        metrics=metrics,
        precision=precision_tag,
    )
    print("\n" + formatted)

    # Log
    logger.log({
        f"{model.name}/imagenet/{precision_tag}/top1": metrics.top1_accuracy,
        f"{model.name}/imagenet/{precision_tag}/top5": metrics.top5_accuracy,
    })

    # Log quantization stats
    logger.log({
        f"{model.name}/quantization/{mode}/original_size_mb": q_stats["original_size_mb"],
        f"{model.name}/quantization/{mode}/quantized_size_mb": q_stats["quantized_size_mb"],
        f"{model.name}/quantization/{mode}/compression_ratio": q_stats["compression_ratio"],
        f"{model.name}/quantization/{mode}/size_reduction_pct": q_stats["size_reduction_pct"],
    })

    # Save
    checkpoint_manager.save_metrics(
        metrics=metrics.to_dict(),
        model_name=model.name,
        dataset_name="imagenet",
        precision=precision_tag,
    )

    # Free quantized model
    del q_backbone
    torch.cuda.empty_cache()

    return metrics, q_stats


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
    text_logger.info(f"Device: {config.device}")
    text_logger.info(f"Debug mode: {config.debug}")
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(config) if config.logging.wandb.enabled else None
    metrics_logger = MetricsLogger(config, wandb_logger)
    
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
        if ds_name == "coco":
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
                        logger=metrics_logger,
                        text_logger=text_logger,
                    )
                    fp32_results[dataset_name] = metrics
                    all_results[model_config.name][dataset_name] = {"fp32": metrics}
                    
                elif dataset_name == "imagenet_c":
                    metrics = evaluate_model_on_imagenet_c(
                        model=model,
                        config=config,
                        checkpoint_manager=checkpoint_manager,
                        logger=metrics_logger,
                        text_logger=text_logger,
                    )
                    fp32_results[dataset_name] = metrics
                    all_results[model_config.name][dataset_name] = {"fp32": metrics}
                
                elif dataset_name == "coco":
                    metrics = evaluate_model_on_coco(
                        model=model,
                        config=config,
                        checkpoint_manager=checkpoint_manager,
                        logger=metrics_logger,
                        text_logger=text_logger,
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
            except Exception as e:
                text_logger.error(f"  [!] Error evaluating on {dataset_name}: {e}")
                raise
        
        # ── Phase 2: Quantized Evaluation ──
        if quant_enabled and quant_modes:
            for qmode in quant_modes:
                text_logger.info(f"\n  ── INT8 {qmode} quantization ──")

                if model.task != "classification":
                    text_logger.info(
                        f"  Skipping quantization for {model_config.name} "
                        f"(task={model.task}, classification only for now)"
                    )
                    continue

                try:
                    q_metrics, q_stats = evaluate_quantized_classification(
                        model=model,
                        config=config,
                        checkpoint_manager=checkpoint_manager,
                        logger=metrics_logger,
                        text_logger=text_logger,
                        mode=qmode,
                    )

                    # Store results
                    prec_key = f"int8_{qmode}"
                    if "imagenet" in all_results[model_config.name]:
                        all_results[model_config.name]["imagenet"][prec_key] = q_metrics
                    else:
                        all_results[model_config.name]["imagenet"] = {prec_key: q_metrics}

                    # Print comparison
                    fp32_m = fp32_results.get("imagenet")
                    if fp32_m is not None:
                        comparison = format_comparison_row(
                            model_name=model_config.name,
                            dataset_name="imagenet",
                            task="classification",
                            fp32_metrics=fp32_m,
                            quant_metrics=q_metrics,
                            quant_mode=qmode,
                            quant_stats=q_stats,
                        )
                        print("\n" + comparison)

                except Exception as e:
                    text_logger.error(
                        f"  [!] Error during INT8 {qmode} for {model_config.name}: {e}"
                    )
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Clear model from memory
        del model
        torch.cuda.empty_cache()
    
    # Log final summary using unified formatting
    # Flatten nested results for backward-compat with format_final_summary
    flat_results = {}
    for mname, ds_dict in all_results.items():
        flat_results[mname] = {}
        for ds_name, prec_dict in ds_dict.items():
            if isinstance(prec_dict, dict) and "fp32" in prec_dict:
                # New nested format – use fp32 for top-level summary
                flat_results[mname][ds_name] = prec_dict["fp32"]
            else:
                flat_results[mname][ds_name] = prec_dict
    
    summary = format_final_summary(flat_results, task_types)
    text_logger.info(summary)
    
    # Finish wandb
    if wandb_logger:
        wandb_logger.finish()
    
    text_logger.info("Experiment completed successfully!")
    
    return all_results


if __name__ == "__main__":
    main()