"""
Model factory for creating and registering models.

This module provides a registry-based factory pattern for
creating models by name.
"""

from typing import Any, Callable, Dict, Optional, Type

import torch.nn as nn

from src.models.base import BaseModel, ModelWrapper
from src.utils.config import ModelConfig
from src.utils.logging import get_logger

logger = get_logger("qda.models.factory")


# Global model registry
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}
_MODEL_BUILDERS: Dict[str, Callable[..., BaseModel]] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model class.
    
    Usage:
        @register_model("resnet50")
        class ResNet50Model(BaseModel):
            ...
    
    Args:
        name: Model name for registration.
        
    Returns:
        Decorator function.
    """
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        if name in _MODEL_REGISTRY:
            logger.warning(f"Model '{name}' already registered. Overwriting.")
        _MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered model: {name}")
        return cls
    return decorator


def register_model_builder(name: str) -> Callable:
    """
    Decorator to register a model builder function.
    
    Usage:
        @register_model_builder("custom_resnet")
        def build_custom_resnet(**kwargs) -> BaseModel:
            ...
    
    Args:
        name: Model name for registration.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable[..., BaseModel]) -> Callable[..., BaseModel]:
        if name in _MODEL_BUILDERS:
            logger.warning(f"Model builder '{name}' already registered. Overwriting.")
        _MODEL_BUILDERS[name] = func
        logger.debug(f"Registered model builder: {name}")
        return func
    return decorator


class ModelFactory:
    """
    Factory class for creating models.
    
    Provides a unified interface for creating models by name,
    with support for custom configurations.
    """
    
    @staticmethod
    def create(
        config: ModelConfig,
        device: str = "cuda",
    ) -> BaseModel:
        """
        Create a model from configuration.
        
        Args:
            config: Model configuration.
            device: Device to load the model on.
            
        Returns:
            Instantiated model.
        """
        architecture = config.architecture.lower()
        
        # Check if we have a registered model class
        if config.name in _MODEL_REGISTRY:
            model_cls = _MODEL_REGISTRY[config.name]
            model = model_cls(
                weights=config.weights,
                num_classes=config.num_classes,
            )
        # Check if we have a registered builder
        elif config.name in _MODEL_BUILDERS:
            builder = _MODEL_BUILDERS[config.name]
            model = builder(
                architecture=config.architecture,
                weights=config.weights,
                num_classes=config.num_classes,
            )
        # Try to create from architecture name
        elif architecture in _MODEL_REGISTRY:
            model_cls = _MODEL_REGISTRY[architecture]
            model = model_cls(
                weights=config.weights,
                num_classes=config.num_classes,
            )
        # Check if it's a detection model
        elif config.task == "detection":
            model = ModelFactory._create_detection_model(config)
        else:
            # Fallback: try to load from torchvision
            model = ModelFactory._create_from_torchvision(config)
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.set_eval_mode()
        
        logger.info(
            f"Created model: {config.name} ({config.architecture}), "
            f"weights={config.weights}, device={device}"
        )
        
        return model
    
    @staticmethod
    def _create_from_torchvision(config: ModelConfig) -> BaseModel:
        """
        Create a model from torchvision.
        
        Args:
            config: Model configuration.
            
        Returns:
            Wrapped torchvision model.
        """
        import torchvision.models as models
        
        architecture = config.architecture.lower()
        
        # Map common names to torchvision functions
        model_map = {
            "resnet50": (models.resnet50, models.ResNet50_Weights),
            "resnet101": (models.resnet101, models.ResNet101_Weights),
            "resnet152": (models.resnet152, models.ResNet152_Weights),
            "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights),
            "mobilenetv2": (models.mobilenet_v2, models.MobileNet_V2_Weights),
            "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights),
            "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights),
            "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights),
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
            "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights),
        }
        
        if architecture not in model_map:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Available: {list(model_map.keys())}"
            )
        
        model_fn, weights_cls = model_map[architecture]
        
        # Get weights
        if config.weights:
            weights = getattr(weights_cls, config.weights, weights_cls.DEFAULT)
        else:
            weights = weights_cls.DEFAULT
        
        # Create model
        model = model_fn(weights=weights)
        
        # Get preprocessing config from weights
        preprocess_config = {
            "input_size": 224,
            "mean": list(weights.transforms().mean),
            "std": list(weights.transforms().std),
        }
        
        # Wrap in our interface
        return ModelWrapper(
            model=model,
            name=config.name,
            task=config.task,
            num_classes=config.num_classes,
            preprocessing_config=preprocess_config,
        )
    
    @staticmethod
    def _create_detection_model(config: ModelConfig) -> BaseModel:
        """
        Create a detection model from torchvision.
        
        Args:
            config: Model configuration.
            
        Returns:
            Detection model instance.
        """
        import torchvision.models.detection as detection_models
        from src.models.detection import DetectionModel
        
        architecture = config.architecture.lower()
        
        # Map architecture names to torchvision functions and weights
        detection_map = {
            "fasterrcnn_resnet50_fpn": (
                detection_models.fasterrcnn_resnet50_fpn,
                detection_models.FasterRCNN_ResNet50_FPN_Weights,
            ),
            "fasterrcnn_resnet50_fpn_v2": (
                detection_models.fasterrcnn_resnet50_fpn_v2,
                detection_models.FasterRCNN_ResNet50_FPN_V2_Weights,
            ),
            "fasterrcnn_mobilenet_v3_large_fpn": (
                detection_models.fasterrcnn_mobilenet_v3_large_fpn,
                detection_models.FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            ),
            "retinanet_resnet50_fpn": (
                detection_models.retinanet_resnet50_fpn,
                detection_models.RetinaNet_ResNet50_FPN_Weights,
            ),
            "retinanet_resnet50_fpn_v2": (
                detection_models.retinanet_resnet50_fpn_v2,
                detection_models.RetinaNet_ResNet50_FPN_V2_Weights,
            ),
            "fcos_resnet50_fpn": (
                detection_models.fcos_resnet50_fpn,
                detection_models.FCOS_ResNet50_FPN_Weights,
            ),
            "ssd300_vgg16": (
                detection_models.ssd300_vgg16,
                detection_models.SSD300_VGG16_Weights,
            ),
            "ssdlite320_mobilenet_v3_large": (
                detection_models.ssdlite320_mobilenet_v3_large,
                detection_models.SSDLite320_MobileNet_V3_Large_Weights,
            ),
        }
        
        if architecture not in detection_map:
            raise ValueError(
                f"Unknown detection architecture: {architecture}. "
                f"Available: {list(detection_map.keys())}"
            )
        
        model_fn, weights_cls = detection_map[architecture]
        
        # Get weights
        if config.weights and config.weights != "None":
            weights = getattr(weights_cls, config.weights, weights_cls.DEFAULT)
        else:
            weights = weights_cls.DEFAULT
        
        # Create model
        backbone = model_fn(weights=weights)
        
        # Create wrapped detection model
        return DetectionModel(
            name=config.name,
            backbone=backbone,
            num_classes=config.num_classes,
            preprocessing_config={
                "min_size": 800,
                "max_size": 1333,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        )
    
    @staticmethod
    def list_available() -> Dict[str, list]:
        """
        List all available models.
        
        Returns:
            Dictionary with 'registered' and 'torchvision' models.
        """
        import torchvision.models as models
        
        # Get registered models
        registered = list(_MODEL_REGISTRY.keys()) + list(_MODEL_BUILDERS.keys())
        
        # Get torchvision models (common ones)
        torchvision_models = [
            "resnet50", "resnet101", "resnet152",
            "mobilenet_v2",
            "vit_b_16", "vit_b_32", "vit_l_16",
            "efficientnet_b0", "efficientnet_b7",
        ]
        
        return {
            "registered": registered,
            "torchvision": torchvision_models,
        }


def get_model(
    name: str,
    architecture: Optional[str] = None,
    weights: str = "DEFAULT",
    task: str = "classification",
    num_classes: int = 1000,
    device: str = "cuda",
) -> BaseModel:
    """
    Convenience function to create a model.
    
    Args:
        name: Model name.
        architecture: Architecture name (defaults to name).
        weights: Pretrained weights name.
        task: Task type.
        num_classes: Number of classes.
        device: Device to load model on.
        
    Returns:
        Instantiated model.
    """
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
    """
    Load model weights from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        device: Device to load model on.
        
    Returns:
        Model with loaded weights.
    """
    import torch
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"Loaded model weights from: {checkpoint_path}")
    
    return model.to(device)
