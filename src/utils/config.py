"""
Configuration management using dataclasses and YAML.

This module provides a type-safe configuration system that loads
experiment settings from YAML files and validates them.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    architecture: str
    weights: str
    task: str = "classification"
    num_classes: int = 1000
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(
            name=data["name"],
            architecture=data["architecture"],
            weights=data["weights"],
            task=data.get("task", "classification"),
            num_classes=data.get("num_classes", 1000),
        )


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    root: str
    batch_size: int = 64
    shuffle: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    split: Optional[str] = None
    
    # ImageNet-C specific
    corruptions: Optional[List[str]] = None
    severities: Optional[List[int]] = None
    
    # COCO specific
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "DatasetConfig":
        """Create DatasetConfig from dictionary."""
        return cls(
            name=data.get("name", name),
            root=data["root"],
            batch_size=data.get("batch_size", 64),
            shuffle=data.get("shuffle", False),
            num_workers=data.get("num_workers", 8),
            pin_memory=data.get("pin_memory", True),
            split=data.get("split"),
            corruptions=data.get("corruptions"),
            severities=data.get("severities"),
            min_size=data.get("min_size"),
            max_size=data.get("max_size"),
        )


@dataclass
class QuantizationConfig:
    """Configuration for quantization workflows."""
    enabled: bool = False
    modes: List[str] = field(default_factory=lambda: ["weight_only"])
    # YOLO export-based quantization options
    yolo_format: Optional[str] = None
    yolo_data: Optional[str] = None
    yolo_fraction: float = 1.0
    yolo_imgsz: int = 640
    yolo_batch: int = 8
    yolo_export_dir: Optional[str] = None
    reuse_yolo_export: bool = True
    # If true, skip FP32 phase runs and evaluate only quantized phases.
    skip_fp32_phase: bool = False
    # Output-level FP32-vs-quantized drift analysis for detection tasks.
    compute_detection_drift: bool = True
    # If false, do not log TensorRT layer-coverage diagnostics to wandb.
    log_coverage_metrics: bool = True
    # Optional explicit calibration budgeting for YOLO export INT8.
    # Precedence in main.py: num_calibration_batches > calibration_num_samples > yolo_fraction.
    num_calibration_batches: Optional[int] = None
    calibration_num_samples: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        """Create QuantizationConfig from dictionary."""
        modes = data.get("modes", ["weight_only"])
        if isinstance(modes, str):
            modes = [modes]
        calibration = data.get("calibration") or {}
        num_calib_batches = data.get("num_calibration_batches")
        calib_num_samples = calibration.get("num_samples")
        return cls(
            enabled=data.get("enabled", False),
            modes=modes,
            yolo_format=data.get("yolo_format"),
            yolo_data=data.get("yolo_data"),
            yolo_fraction=float(data.get("yolo_fraction", 1.0)),
            yolo_imgsz=int(data.get("yolo_imgsz", 640)),
            yolo_batch=int(data.get("yolo_batch", 8)),
            yolo_export_dir=data.get("yolo_export_dir"),
            reuse_yolo_export=bool(data.get("reuse_yolo_export", True)),
            skip_fp32_phase=bool(data.get("skip_fp32_phase", False)),
            compute_detection_drift=bool(data.get("compute_detection_drift", True)),
            log_coverage_metrics=bool(data.get("log_coverage_metrics", True)),
            num_calibration_batches=(
                int(num_calib_batches) if num_calib_batches is not None else None
            ),
            calibration_num_samples=(
                int(calib_num_samples) if calib_num_samples is not None else None
            ),
        )


@dataclass
class RectiQConfig:
    """Configuration for Recti-Q adapter training on frozen quantized detection models."""
    enabled: bool = False
    # If true, run only Recti-Q phase logging for YOLO detection (skip FP32/INT8 wandb runs).
    only: bool = False
    # If true, skip base quantized-model evaluation before Recti-Q training/eval.
    skip_base_quant_eval: bool = False
    rank: int = 8
    rank_per_scale: Optional[List[int]] = None
    alpha: float = 16.0
    alpha_per_scale: Optional[List[float]] = None
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    feature_kd_weight: float = 1.0
    residual_reg_weight: float = 1e-4
    # Detection supervision weight (native YOLO loss in Recti-Q training loop).
    task_loss_weight: float = 1.0
    # Where to attach adapters:
    # - detect_input: final Detect-input tensors
    # - neck_pre_detect: one block earlier inside the neck
    adapter_target: str = "detect_input"
    use_teacher: bool = True
    train_split: Optional[str] = None
    val_split: Optional[str] = None
    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    max_batches_per_epoch: Optional[int] = None
    imgsz: Optional[int] = None
    output_dir: Optional[str] = None
    # Recti-Q student backend:
    # - "ptq_surrogate": trainable PyTorch student with fixed INT8 fake-quant (PTQ-style)
    # - "runtime_export": use exported backend directly (non-trainable for engine/openvino)
    student_backend: str = "ptq_surrogate"
    # Calibration batches for fixed PTQ surrogate scale estimation.
    ptq_calibration_batches: Optional[int] = None
    # Use straight-through estimator through fixed Q/DQ during LoRA training.
    ptq_use_ste: bool = True
    # Keep corrected features on quantized manifold: y=QDQ(QDQ(x)+delta).
    requantize_after_residual: bool = True
    # Adapter capacity options.
    adapter_use_dwconv: bool = True
    adapter_dropout: float = 0.0
    # Optional selective unfreezing for detect classification branch.
    unfreeze_detect_cv3: bool = False
    cv3_lr: Optional[float] = None
    # Optional scale recalibration after warmup epochs.
    recalibration_epoch: Optional[int] = None
    recalibration_batches: Optional[int] = None
    # Optional pre-training closeness report: runtime-export INT8 vs PTQ surrogate.
    compare_to_runtime_export: bool = True
    compare_max_batches: Optional[int] = 20
    # If true, run validation only on the final epoch.
    val_final_only: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RectiQConfig":
        """Create RectiQConfig from dictionary."""
        max_batches = data.get("max_batches_per_epoch")
        train_bs = data.get("train_batch_size")
        val_bs = data.get("val_batch_size")
        imgsz = data.get("imgsz")
        ptq_calib_batches = data.get("ptq_calibration_batches")
        rank_per_scale = data.get("rank_per_scale")
        alpha_per_scale = data.get("alpha_per_scale")
        cv3_lr = data.get("cv3_lr")
        recalib_epoch = data.get("recalibration_epoch")
        recalib_batches = data.get("recalibration_batches")
        compare_max_batches = data.get("compare_max_batches")
        return cls(
            enabled=bool(data.get("enabled", False)),
            only=bool(data.get("only", data.get("rectiq_only", False))),
            skip_base_quant_eval=bool(
                data.get("skip_base_quant_eval", data.get("skip_quant_eval", False))
            ),
            rank=int(data.get("rank", 8)),
            rank_per_scale=(
                [int(v) for v in rank_per_scale]
                if isinstance(rank_per_scale, list) and len(rank_per_scale) > 0
                else None
            ),
            alpha=float(data.get("alpha", 16.0)),
            alpha_per_scale=(
                [float(v) for v in alpha_per_scale]
                if isinstance(alpha_per_scale, list) and len(alpha_per_scale) > 0
                else None
            ),
            epochs=int(data.get("epochs", 3)),
            lr=float(data.get("lr", 3e-4)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            feature_kd_weight=float(data.get("feature_kd_weight", 1.0)),
            residual_reg_weight=float(data.get("residual_reg_weight", 1e-4)),
            task_loss_weight=float(data.get("task_loss_weight", 1.0)),
            adapter_target=str(data.get("adapter_target", "detect_input")),
            use_teacher=bool(data.get("use_teacher", True)),
            train_split=data.get("train_split"),
            val_split=data.get("val_split"),
            train_batch_size=(int(train_bs) if train_bs is not None else None),
            val_batch_size=(int(val_bs) if val_bs is not None else None),
            num_workers=int(data.get("num_workers", 0)),
            max_batches_per_epoch=(int(max_batches) if max_batches is not None else None),
            imgsz=(int(imgsz) if imgsz is not None else None),
            output_dir=data.get("output_dir"),
            student_backend=str(data.get("student_backend", "ptq_surrogate")),
            ptq_calibration_batches=(
                int(ptq_calib_batches) if ptq_calib_batches is not None else None
            ),
            ptq_use_ste=bool(data.get("ptq_use_ste", True)),
            requantize_after_residual=bool(data.get("requantize_after_residual", True)),
            adapter_use_dwconv=bool(data.get("adapter_use_dwconv", True)),
            adapter_dropout=float(data.get("adapter_dropout", 0.0)),
            unfreeze_detect_cv3=bool(data.get("unfreeze_detect_cv3", False)),
            cv3_lr=(float(cv3_lr) if cv3_lr is not None else None),
            recalibration_epoch=(int(recalib_epoch) if recalib_epoch is not None else None),
            recalibration_batches=(
                int(recalib_batches) if recalib_batches is not None else None
            ),
            compare_to_runtime_export=bool(data.get("compare_to_runtime_export", True)),
            compare_max_batches=(
                int(compare_max_batches) if compare_max_batches is not None else None
            ),
            val_final_only=bool(data.get("val_final_only", False)),
        )


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = True
    project: str = "quantization-decision-analysis"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    # If false, do not log per-run model_size_mb metrics.
    log_model_size: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WandbConfig":
        """Create WandbConfig from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            project=data.get("project", "quantization-decision-analysis"),
            entity=data.get("entity"),
            tags=data.get("tags", []),
            notes=data.get("notes"),
            log_model_size=bool(data.get("log_model_size", True)),
        )


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "./logs"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggingConfig":
        """Create LoggingConfig from dictionary."""
        wandb_data = data.get("wandb", {})
        return cls(
            level=data.get("level", "INFO"),
            log_to_file=data.get("log_to_file", True),
            log_dir=data.get("log_dir", "./logs"),
            wandb=WandbConfig.from_dict(wandb_data),
        )


@dataclass
class OutputConfig:
    """Configuration for output and checkpointing."""
    save_predictions: bool = True
    prediction_format: str = "pickle"  # pickle, json, hdf5
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    save_logits: bool = True
    save_confidence: bool = True
    save_boxes: bool = False  # For detection
    save_scores: bool = False  # For detection

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        """Create OutputConfig from dictionary."""
        return cls(
            save_predictions=data.get("save_predictions", True),
            prediction_format=data.get("prediction_format", "pickle"),
            checkpoint_dir=data.get("checkpoint_dir", "./checkpoints"),
            results_dir=data.get("results_dir", "./results"),
            save_logits=data.get("save_logits", True),
            save_confidence=data.get("save_confidence", True),
            save_boxes=data.get("save_boxes", False),
            save_scores=data.get("save_scores", False),
        )


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    name: str
    description: str = ""
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 8
    debug: bool = False
    debug_samples: int = 100
    
    models: List[ModelConfig] = field(default_factory=list)
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    rectiq: RectiQConfig = field(default_factory=RectiQConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create ExperimentConfig from dictionary."""
        exp_data = data.get("experiment", {})
        
        # Parse models
        models = [
            ModelConfig.from_dict(m) 
            for m in data.get("models", [])
        ]
        
        # Parse datasets
        datasets = {
            name: DatasetConfig.from_dict(name, cfg)
            for name, cfg in data.get("datasets", {}).items()
        }
        
        return cls(
            name=exp_data.get("name", "unnamed_experiment"),
            description=exp_data.get("description", ""),
            seed=exp_data.get("seed", 42),
            device=exp_data.get("device", "cuda"),
            num_workers=exp_data.get("num_workers", 8),
            debug=exp_data.get("debug", False),
            debug_samples=exp_data.get("debug_samples", 100),
            models=models,
            datasets=datasets,
            quantization=QuantizationConfig.from_dict(
                data.get("quantization", {})
            ),
            rectiq=RectiQConfig.from_dict(data.get("rectiq", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
        )
    
    def get_dataset(self, name: str) -> DatasetConfig:
        """Get dataset configuration by name."""
        if name not in self.datasets:
            available = list(self.datasets.keys())
            raise KeyError(
                f"Dataset '{name}' not found. Available: {available}"
            )
        return self.datasets[name]
    
    def get_model(self, name: str) -> ModelConfig:
        """Get model configuration by name."""
        for model in self.models:
            if model.name == name:
                return model
        available = [m.name for m in self.models]
        raise KeyError(f"Model '{name}' not found. Available: {available}")


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        ExperimentConfig object with all settings.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    return ExperimentConfig.from_dict(raw_config)


def save_config(config: ExperimentConfig, output_path: Union[str, Path]) -> None:
    """
    Save experiment configuration to a YAML file.
    
    Args:
        config: ExperimentConfig object to save.
        output_path: Path where to save the configuration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict (simplified version)
    # For a full implementation, use dataclasses.asdict with custom handling
    with open(output_path, "w") as f:
        yaml.dump({"experiment": {"name": config.name}}, f)


def merge_configs(
    base: ExperimentConfig, 
    overrides: Dict[str, Any]
) -> ExperimentConfig:
    """
    Merge override values into a base configuration.
    
    Useful for command-line argument overrides.
    
    Args:
        base: Base ExperimentConfig.
        overrides: Dictionary of values to override.
        
    Returns:
        New ExperimentConfig with overrides applied.
    """
    # Convert base to dict, apply overrides, convert back
    # This is a simplified implementation
    import copy
    new_config = copy.deepcopy(base)
    
    for key, value in overrides.items():
        if hasattr(new_config, key):
            setattr(new_config, key, value)
    
    return new_config
