"""
Model modules for loading and managing neural networks.
"""

from src.models.base import BaseModel, ModelOutput
from src.models.factory import ModelFactory, get_model, register_model
from src.models.classification import (
    ClassificationModel,
    ResNet50Model,
    MobileNetV2Model,
    ViTBaseModel,
)
from src.models.detection import (
    DetectionModel,
    FasterRCNNResNet50,
    RetinaNetResNet50,
    FCOSResNet50,
    SSD300VGG16,
    YOLODetectionModel,
    YOLOv8Nano,
    YOLOv8Small,
    YOLOv8Medium,
    YOLOv8Large,
    YOLOv8XLarge,
)

__all__ = [
    "BaseModel",
    "ModelOutput",
    "ModelFactory",
    "get_model",
    "register_model",
    "ClassificationModel",
    "ResNet50Model",
    "MobileNetV2Model",
    "ViTBaseModel",
    "DetectionModel",
    "FasterRCNNResNet50",
    "RetinaNetResNet50",
    "FCOSResNet50",
    "SSD300VGG16",
    "YOLODetectionModel",
    "YOLOv8Nano",
    "YOLOv8Small",
    "YOLOv8Medium",
    "YOLOv8Large",
    "YOLOv8XLarge",
]
