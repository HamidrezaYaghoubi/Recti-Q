"""
Classification models: ResNet, MobileNet, ViT.

This module provides concrete implementations of classification
models with the BaseModel interface.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.base import BaseModel, ModelOutput
from src.models.factory import register_model
from src.utils.logging import get_logger

logger = get_logger("qda.models.classification")


class ClassificationModel(BaseModel):
    """
    Base class for classification models.
    
    Provides common functionality for all classification models.
    """
    
    def __init__(
        self,
        name: str,
        backbone: nn.Module,
        num_classes: int = 1000,
        preprocessing_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize classification model.
        
        Args:
            name: Model name.
            backbone: The backbone neural network.
            num_classes: Number of output classes.
            preprocessing_config: Preprocessing configuration.
        """
        super().__init__(name, task="classification", num_classes=num_classes)
        self.backbone = backbone
        self._preprocessing_config = preprocessing_config or {
            "input_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> ModelOutput:
        """
        Run inference and return structured output.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            ModelOutput with predictions, logits, probabilities, and confidences.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = probabilities.max(dim=-1)
            
            return ModelOutput(
                predictions=predictions,
                logits=logits,
                probabilities=probabilities,
                confidences=confidences,
            )
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._preprocessing_config
    
    def get_top_k_predictions(
        self, 
        x: torch.Tensor, 
        k: int = 5,
    ) -> tuple:
        """
        Get top-k predictions for input.
        
        Args:
            x: Input tensor.
            k: Number of top predictions to return.
            
        Returns:
            Tuple of (top_k_probs, top_k_indices).
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = probabilities.topk(k, dim=-1)
            return top_k_probs, top_k_indices


@register_model("resnet50")
class ResNet50Model(ClassificationModel):
    """
    ResNet-50 model for classification.
    """
    
    def __init__(
        self,
        weights: str = "IMAGENET1K_V2",
        num_classes: int = 1000,
    ):
        """
        Initialize ResNet-50.
        
        Args:
            weights: Pretrained weights name.
            num_classes: Number of output classes.
        """
        # Get weights
        if weights == "IMAGENET1K_V2":
            weights_obj = models.ResNet50_Weights.IMAGENET1K_V2
        elif weights == "IMAGENET1K_V1":
            weights_obj = models.ResNet50_Weights.IMAGENET1K_V1
        elif weights == "DEFAULT":
            weights_obj = models.ResNet50_Weights.DEFAULT
        else:
            weights_obj = None
        
        # Create backbone
        backbone = models.resnet50(weights=weights_obj)
        
        # Get preprocessing from weights
        if weights_obj is not None:
            transforms = weights_obj.transforms()
            preprocess_config = {
                "input_size": 224,
                "mean": list(transforms.mean),
                "std": list(transforms.std),
            }
        else:
            preprocess_config = None
        
        super().__init__(
            name="resnet50",
            backbone=backbone,
            num_classes=num_classes,
            preprocessing_config=preprocess_config,
        )
        
        logger.info(f"Initialized ResNet50 with weights: {weights}")


@register_model("resnet101")
class ResNet101Model(ClassificationModel):
    """ResNet-101 model for classification."""
    
    def __init__(
        self,
        weights: str = "IMAGENET1K_V2",
        num_classes: int = 1000,
    ):
        if weights == "IMAGENET1K_V2":
            weights_obj = models.ResNet101_Weights.IMAGENET1K_V2
        elif weights == "IMAGENET1K_V1":
            weights_obj = models.ResNet101_Weights.IMAGENET1K_V1
        else:
            weights_obj = models.ResNet101_Weights.DEFAULT
        
        backbone = models.resnet101(weights=weights_obj)
        
        transforms = weights_obj.transforms()
        preprocess_config = {
            "input_size": 224,
            "mean": list(transforms.mean),
            "std": list(transforms.std),
        }
        
        super().__init__(
            name="resnet101",
            backbone=backbone,
            num_classes=num_classes,
            preprocessing_config=preprocess_config,
        )


@register_model("mobilenetv2")
@register_model("mobilenet_v2")
class MobileNetV2Model(ClassificationModel):
    """
    MobileNetV2 model for classification.
    
    Efficient model suitable for mobile/edge deployment.
    """
    
    def __init__(
        self,
        weights: str = "IMAGENET1K_V1",
        num_classes: int = 1000,
    ):
        """
        Initialize MobileNetV2.
        
        Args:
            weights: Pretrained weights name.
            num_classes: Number of output classes.
        """
        if weights == "IMAGENET1K_V2":
            weights_obj = models.MobileNet_V2_Weights.IMAGENET1K_V2
        elif weights == "IMAGENET1K_V1":
            weights_obj = models.MobileNet_V2_Weights.IMAGENET1K_V1
        else:
            weights_obj = models.MobileNet_V2_Weights.DEFAULT
        
        backbone = models.mobilenet_v2(weights=weights_obj)
        
        transforms = weights_obj.transforms()
        preprocess_config = {
            "input_size": 224,
            "mean": list(transforms.mean),
            "std": list(transforms.std),
        }
        
        super().__init__(
            name="mobilenetv2",
            backbone=backbone,
            num_classes=num_classes,
            preprocessing_config=preprocess_config,
        )
        
        logger.info(f"Initialized MobileNetV2 with weights: {weights}")


@register_model("vit_base")
@register_model("vit_b_16")
class ViTBaseModel(ClassificationModel):
    """
    Vision Transformer (ViT-B/16) model for classification.
    """
    
    def __init__(
        self,
        weights: str = "IMAGENET1K_V1",
        num_classes: int = 1000,
    ):
        """
        Initialize ViT-Base/16.
        
        Args:
            weights: Pretrained weights name.
            num_classes: Number of output classes.
        """
        if weights == "IMAGENET1K_SWAG_E2E_V1":
            weights_obj = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        elif weights == "IMAGENET1K_SWAG_LINEAR_V1":
            weights_obj = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        elif weights == "IMAGENET1K_V1":
            weights_obj = models.ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights_obj = models.ViT_B_16_Weights.DEFAULT
        
        backbone = models.vit_b_16(weights=weights_obj)
        
        transforms = weights_obj.transforms()
        preprocess_config = {
            "input_size": 224,
            "mean": list(transforms.mean),
            "std": list(transforms.std),
        }
        
        super().__init__(
            name="vit_base",
            backbone=backbone,
            num_classes=num_classes,
            preprocessing_config=preprocess_config,
        )
        
        logger.info(f"Initialized ViT-Base/16 with weights: {weights}")


@register_model("vit_b_32")
class ViTBase32Model(ClassificationModel):
    """Vision Transformer (ViT-B/32) model for classification."""
    
    def __init__(
        self,
        weights: str = "IMAGENET1K_V1",
        num_classes: int = 1000,
    ):
        weights_obj = models.ViT_B_32_Weights.IMAGENET1K_V1
        backbone = models.vit_b_32(weights=weights_obj)
        
        transforms = weights_obj.transforms()
        preprocess_config = {
            "input_size": 224,
            "mean": list(transforms.mean),
            "std": list(transforms.std),
        }
        
        super().__init__(
            name="vit_b_32",
            backbone=backbone,
            num_classes=num_classes,
            preprocessing_config=preprocess_config,
        )


@register_model("efficientnet_b0")
class EfficientNetB0Model(ClassificationModel):
    """EfficientNet-B0 model for classification."""
    
    def __init__(
        self,
        weights: str = "IMAGENET1K_V1",
        num_classes: int = 1000,
    ):
        weights_obj = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights_obj)
        
        transforms = weights_obj.transforms()
        preprocess_config = {
            "input_size": 224,
            "mean": list(transforms.mean),
            "std": list(transforms.std),
        }
        
        super().__init__(
            name="efficientnet_b0",
            backbone=backbone,
            num_classes=num_classes,
            preprocessing_config=preprocess_config,
        )


# TODO: Week 2-3 - Add detection models
# @register_model("fasterrcnn_resnet50")
# class FasterRCNNResNet50(BaseModel):
#     """Faster R-CNN with ResNet-50 backbone for object detection."""
#     pass
