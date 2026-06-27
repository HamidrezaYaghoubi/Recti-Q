"""
Configuration management using dataclasses and YAML.

Loads experiment settings from a YAML file into typed dataclasses. This is the
shared contract between the model, dataset, quantization, and Recti-Q modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ModelConfig:
    """Configuration for a single classification model."""
    name: str
    architecture: str
    weights: str = "pretrained"
    task: str = "classification"
    num_classes: int = 1000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=data["name"],
            architecture=data["architecture"],
            weights=data.get("weights", "pretrained"),
            task=data.get("task", "classification"),
            num_classes=data.get("num_classes", 1000),
        )


@dataclass
class DatasetConfig:
    """Configuration for a dataset (ImageNet / ImageNet-C / PACS)."""
    name: str
    root: str
    batch_size: int = 256
    shuffle: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    split: Optional[str] = None

    # ImageNet-C specific
    corruptions: Optional[List[str]] = None
    severities: Optional[List[int]] = None

    # ImageNet-C source-training subsample (5% class-balanced ImageNet-1k train)
    train_root: Optional[str] = None
    subset_fraction: float = 0.05
    subset_balanced: bool = True
    subset_seed: int = 42

    # PACS leave-one-domain-out
    target_domain: Optional[str] = None
    domains: Optional[List[str]] = None

    # PACS base model: short ERM fine-tune on source domains before quantization.
    # (ImageNet-C uses the pretrained model directly, so this stays 0.)
    base_train_epochs: int = 0
    base_lr: float = 1e-4

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "DatasetConfig":
        return cls(
            name=data.get("name", name),
            root=data["root"],
            batch_size=data.get("batch_size", 256),
            shuffle=data.get("shuffle", False),
            num_workers=data.get("num_workers", 8),
            pin_memory=data.get("pin_memory", True),
            split=data.get("split"),
            corruptions=data.get("corruptions"),
            severities=data.get("severities"),
            train_root=data.get("train_root"),
            subset_fraction=float(data.get("subset_fraction", 0.05)),
            subset_balanced=bool(data.get("subset_balanced", True)),
            subset_seed=int(data.get("subset_seed", 42)),
            target_domain=data.get("target_domain"),
            domains=data.get("domains"),
            base_train_epochs=int(data.get("base_train_epochs", 0)),
            base_lr=float(data.get("base_lr", 1e-4)),
        )


@dataclass
class QuantizationConfig:
    """
    Configuration for PTQ quantization.

    The canonical paper baseline is W4 (4-bit weight-only via torchao
    Int4WeightOnlyConfig with HQQ). Other torchao modes (W8A16, W8A8, ...) are
    kept as optional extras for exploration.
    """
    enabled: bool = True
    modes: List[str] = field(default_factory=lambda: ["W4"])
    group_size: int = 128
    use_hqq: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        modes = data.get("modes", ["W4"])
        if isinstance(modes, str):
            modes = [modes]
        return cls(
            enabled=bool(data.get("enabled", True)),
            modes=modes,
            group_size=int(data.get("group_size", 128)),
            use_hqq=bool(data.get("use_hqq", True)),
        )


@dataclass
class RectiQConfig:
    """
    Configuration for the Recti-Q classifier-head LoRA adapter.

    Implements the paper method: a low-rank adapter on pre-classifier features,
    trained source-only with L = L_CE + kd_lambda * L_KD (KD to a frozen FP32
    teacher at temperature T). Teacher-free when kd_lambda == 0.
    """
    enabled: bool = True
    rank: int = 64
    alpha: float = 16.0
    kd_lambda: float = 1.0
    temperature: float = 4.0
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_batch_size: int = 128
    val_batch_size: int = 256
    num_workers: int = 8
    # "feature": adapt pre-classifier features u (paper); "logit": adapt logits z_q (ablation).
    adapter_space: str = "feature"
    output_dir: Optional[str] = None
    # Debug / quick-run caps (None = full pass).
    max_batches_per_epoch: Optional[int] = None
    val_max_batches: Optional[int] = None

    @property
    def use_teacher(self) -> bool:
        return float(self.kd_lambda) > 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RectiQConfig":
        max_batches = data.get("max_batches_per_epoch")
        val_max = data.get("val_max_batches")
        return cls(
            enabled=bool(data.get("enabled", True)),
            rank=int(data.get("rank", 64)),
            alpha=float(data.get("alpha", 16.0)),
            kd_lambda=float(data.get("kd_lambda", 1.0)),
            temperature=float(data.get("temperature", 4.0)),
            epochs=int(data.get("epochs", 5)),
            lr=float(data.get("lr", 3e-4)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            train_batch_size=int(data.get("train_batch_size", 128)),
            val_batch_size=int(data.get("val_batch_size", 256)),
            num_workers=int(data.get("num_workers", 8)),
            adapter_space=str(data.get("adapter_space", "feature")),
            output_dir=data.get("output_dir"),
            max_batches_per_epoch=(int(max_batches) if max_batches is not None else None),
            val_max_batches=(int(val_max) if val_max is not None else None),
        )


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = True
    project: str = "recti-q"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    log_model_size: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WandbConfig":
        return cls(
            enabled=data.get("enabled", True),
            project=data.get("project", "recti-q"),
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
        return cls(
            level=data.get("level", "INFO"),
            log_to_file=data.get("log_to_file", True),
            log_dir=data.get("log_dir", "./logs"),
            wandb=WandbConfig.from_dict(data.get("wandb", {})),
        )


@dataclass
class OutputConfig:
    """Configuration for output and checkpointing."""
    save_predictions: bool = False
    prediction_format: str = "pickle"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    save_logits: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        return cls(
            save_predictions=data.get("save_predictions", False),
            prediction_format=data.get("prediction_format", "pickle"),
            checkpoint_dir=data.get("checkpoint_dir", "./checkpoints"),
            results_dir=data.get("results_dir", "./results"),
            save_logits=data.get("save_logits", False),
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
        exp_data = data.get("experiment", {})
        models = [ModelConfig.from_dict(m) for m in data.get("models", [])]
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
            quantization=QuantizationConfig.from_dict(data.get("quantization", {})),
            rectiq=RectiQConfig.from_dict(data.get("rectiq", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
        )

    def get_dataset(self, name: str) -> DatasetConfig:
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(self.datasets)}")
        return self.datasets[name]

    def get_model(self, name: str) -> ModelConfig:
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(f"Model '{name}' not found. Available: {[m.name for m in self.models]}")


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    return ExperimentConfig.from_dict(raw_config)
