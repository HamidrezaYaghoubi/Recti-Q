"""
Model factory for Recti-Q – timm-only, classification.

Registry pattern: use @register_model("name") on a builder function or class,
then ModelFactory.create(ModelConfig) produces the model.
"""

from typing import Any, Callable, Dict, Optional, Type

import torch

from src.models.base import BaseModel
from src.utils.config import ModelConfig
from src.utils.logging import get_logger

logger = get_logger("recti_q.models.factory")

# Global registries
_MODEL_REGISTRY: Dict[str, Any] = {}   # name -> class or builder function
_MODEL_BUILDERS: Dict[str, Callable[..., BaseModel]] = {}


def register_model(name: str) -> Callable:
    """Decorator to register a model class or builder function.

    The decorated object is stored and called by ModelFactory.create with
    (weights=..., num_classes=...) keyword arguments.

    Usage:
        @register_model("resnet50")
        def build_resnet50(weights="pretrained", num_classes=1000):
            return TimmClassifier(...)
    """
    def decorator(obj):
        if name in _MODEL_REGISTRY:
            logger.debug(f"register_model: overwriting '{name}'")
        _MODEL_REGISTRY[name] = obj
        logger.debug(f"Registered model: {name}")
        return obj
    return decorator


def register_model_builder(name: str) -> Callable:
    """Decorator to register a model builder function (explicit variant)."""
    def decorator(func: Callable[..., BaseModel]) -> Callable[..., BaseModel]:
        if name in _MODEL_BUILDERS:
            logger.debug(f"register_model_builder: overwriting '{name}'")
        _MODEL_BUILDERS[name] = func
        logger.debug(f"Registered model builder: {name}")
        return func
    return decorator


class ModelFactory:
    """Creates models from ModelConfig using the registered builders."""

    @staticmethod
    def create(config: ModelConfig, device: str = "cuda") -> BaseModel:
        """Build, move to device, set eval, and return a model.

        Looks up config.name first, then config.architecture (and its lower-case
        form) in the registry.
        """
        arch = config.architecture
        arch_lower = arch.lower()

        key = None
        for candidate in (config.name, arch, arch_lower):
            if candidate in _MODEL_REGISTRY:
                key = candidate
                break
            if candidate in _MODEL_BUILDERS:
                key = candidate
                break

        if key is None:
            raise ValueError(
                f"No registered model for name='{config.name}' / "
                f"architecture='{config.architecture}'. "
                f"Available: {list(_MODEL_REGISTRY.keys())}"
            )

        builder = _MODEL_REGISTRY.get(key) or _MODEL_BUILDERS[key]
        model = builder(weights=config.weights, num_classes=config.num_classes)

        model = model.to(device)
        model.set_eval_mode()

        logger.info(
            f"Created model '{config.name}' ({config.architecture}), "
            f"weights={config.weights}, device={device}"
        )
        return model

    @staticmethod
    def list_available() -> Dict[str, list]:
        """Return the names of all registered models."""
        return {
            "registered": sorted(set(list(_MODEL_REGISTRY.keys()) + list(_MODEL_BUILDERS.keys())))
        }


def get_model(
    name: str,
    architecture: Optional[str] = None,
    weights: str = "pretrained",
    task: str = "classification",
    num_classes: int = 1000,
    device: str = "cuda",
) -> BaseModel:
    """Convenience wrapper around ModelFactory.create."""
    config = ModelConfig(
        name=name,
        architecture=architecture or name,
        weights=weights,
        task=task,
        num_classes=num_classes,
    )
    return ModelFactory.create(config, device=device)


def load_model_from_checkpoint(
    checkpoint_path: str,
    model: BaseModel,
    device: str = "cuda",
) -> BaseModel:
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded weights from: {checkpoint_path}")
    return model.to(device)
