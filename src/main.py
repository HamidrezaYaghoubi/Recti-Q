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
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, ExperimentConfig, ModelConfig
from src.utils.logging import setup_logging, get_logger, WandbLogger, MetricsLogger
from src.utils.checkpoint import CheckpointManager
from src.models import ModelFactory, BaseModel
from src.datasets import get_imagenet_loader, get_imagenet_c_loader, get_all_imagenet_c_loaders
from src.evaluation import MetricsComputer, ClassificationMetrics


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
    text_logger.info(f"Evaluating {model.name} on ImageNet...")
    
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
    
    text_logger.info(f"Results: {metrics}")
    
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
    text_logger.info(f"Evaluating {model.name} on ImageNet-C...")
    
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
        text_logger.info(f"  Corruption: {corruption}, Severity: {severity}")
        
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
            description=f"ImageNet-C {corruption}/{severity}",
        )
        
        metrics = results["metrics"]
        all_metrics[(corruption, severity)] = metrics
        
        # Log metrics
        logger.log({
            f"{model.name}/imagenet_c/{corruption}/s{severity}/top1": metrics.top1_accuracy,
            f"{model.name}/imagenet_c/{corruption}/s{severity}/top5": metrics.top5_accuracy,
        })
        
        text_logger.info(f"    Results: {metrics}")
        
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
    
    text_logger.info(f"Mean Top-1: {mean_top1:.2f}%, Mean Top-5: {mean_top5:.2f}%")
    
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
    from src.datasets import get_coco_loader
    from src.evaluation import COCOEvaluator, remap_yolo_to_coco_labels
    from pycocotools.coco import COCO
    
    text_logger.info(f"Evaluating {model.name} on COCO...")
    
    # Get dataset config
    dataset_config = config.get_dataset("coco")
    
    # Load COCO ground truth annotations
    ann_file = Path(dataset_config.root) / "annotations" / "instances_val2017.json"
    text_logger.info(f"Loading COCO annotations from {ann_file}")
    coco_gt = COCO(str(ann_file))
    
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
        text_logger.error(f"COCO dataset not found: {e}")
        text_logger.info("Please download COCO dataset using: ./scripts/download_coco.sh")
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
    text_logger.info("Computing COCO metrics (this may take a moment)...")
    metrics = coco_evaluator.compute()
    
    # Add summary stats
    total_detections = sum(len(p["boxes"]) for p in all_predictions)
    metrics["num_images"] = len(all_predictions)
    metrics["total_detections"] = total_detections
    metrics["avg_detections_per_image"] = total_detections / len(all_predictions) if all_predictions else 0
    
    text_logger.info(f"Results: mAP={metrics['mAP']:.2f}%, mAP50={metrics['mAP50']:.2f}%, mAP75={metrics['mAP75']:.2f}%")
    text_logger.info(f"         AR@100={metrics['AR_100']:.2f}%, num_images={metrics['num_images']}")
    
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
    
    text_logger.info(f"COCO evaluation complete.")
    
    return metrics


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
    
    text_logger.info(f"Models: {[m.name for m in model_configs]}")
    text_logger.info(f"Datasets: {datasets_to_eval}")
    
    # Results storage
    all_results = {}
    
    # Evaluate each model
    for model_config in model_configs:
        text_logger.info(f"\n{'='*60}")
        text_logger.info(f"Evaluating model: {model_config.name}")
        text_logger.info(f"{'='*60}")
        
        # Create model
        model = ModelFactory.create(model_config, device=config.device)
        model_info = model.get_model_info()
        text_logger.info(f"Model info: {model_info}")
        
        all_results[model_config.name] = {}
        
        # Evaluate on each dataset
        for dataset_name in datasets_to_eval:
            if dataset_name not in config.datasets:
                text_logger.warning(f"Dataset {dataset_name} not in config, skipping")
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
                    all_results[model_config.name][dataset_name] = metrics
                    
                elif dataset_name == "imagenet_c":
                    metrics = evaluate_model_on_imagenet_c(
                        model=model,
                        config=config,
                        checkpoint_manager=checkpoint_manager,
                        logger=metrics_logger,
                        text_logger=text_logger,
                    )
                    all_results[model_config.name][dataset_name] = metrics
                
                elif dataset_name == "coco":
                    metrics = evaluate_model_on_coco(
                        model=model,
                        config=config,
                        checkpoint_manager=checkpoint_manager,
                        logger=metrics_logger,
                        text_logger=text_logger,
                    )
                    all_results[model_config.name][dataset_name] = metrics
                    
                else:
                    text_logger.warning(
                        f"Dataset {dataset_name} not yet supported, skipping"
                    )
                    
            except FileNotFoundError as e:
                text_logger.error(f"Dataset not found: {e}")
                continue
            except Exception as e:
                text_logger.error(f"Error evaluating on {dataset_name}: {e}")
                raise
        
        # Clear model from memory
        del model
        torch.cuda.empty_cache()
    
    # Log final summary
    text_logger.info(f"\n{'='*60}")
    text_logger.info("FINAL SUMMARY")
    text_logger.info(f"{'='*60}")
    
    for model_name, model_results in all_results.items():
        text_logger.info(f"\n{model_name}:")
        for dataset_name, metrics in model_results.items():
            if isinstance(metrics, ClassificationMetrics):
                text_logger.info(f"  {dataset_name}: {metrics}")
            elif isinstance(metrics, dict):
                # Check if it's ImageNet-C results (has ClassificationMetrics) or detection results
                first_value = next(iter(metrics.values()), None)
                if isinstance(first_value, ClassificationMetrics):
                    # ImageNet-C results
                    mean_top1 = sum(m.top1_accuracy for m in metrics.values()) / len(metrics)
                    text_logger.info(f"  {dataset_name}: Mean Top-1 = {mean_top1:.2f}%")
                else:
                    # Detection results (dict with num_images, total_detections, etc.)
                    text_logger.info(f"  {dataset_name}: {metrics}")
    
    # Finish wandb
    if wandb_logger:
        wandb_logger.finish()
    
    text_logger.info("\nExperiment completed!")
    
    return all_results


if __name__ == "__main__":
    main()
