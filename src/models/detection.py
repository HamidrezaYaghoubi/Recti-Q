"""
Detection models: FasterRCNN, RetinaNet, etc.

This module provides concrete implementations of object detection
models with the BaseModel interface.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models.detection as detection_models

from src.models.base import BaseModel, ModelOutput
from src.models.factory import register_model
from src.utils.logging import get_logger

logger = get_logger("qda.models.detection")


class DetectionModel(BaseModel):
    """
    Base class for object detection models.
    
    Provides common functionality for all detection models.
    """
    
    def __init__(
        self,
        name: str,
        backbone: nn.Module,
        num_classes: int = 91,  # COCO has 91 classes
        preprocessing_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize detection model.
        
        Args:
            name: Model name.
            backbone: The detection model backbone.
            num_classes: Number of output classes.
            preprocessing_config: Preprocessing configuration.
        """
        super().__init__(name, task="detection", num_classes=num_classes)
        self.backbone = backbone
        self._preprocessing_config = preprocessing_config or {
            "min_size": 800,
            "max_size": 1333,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    
    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass returning detection outputs.
        
        For detection models, the output is a list of dictionaries,
        one per image in the batch.
        """
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> ModelOutput:
        """
        Run inference and return structured output.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or list of tensors.
            
        Returns:
            ModelOutput with detection predictions.
        """
        with torch.no_grad():
            # Detection models expect a list of images
            if isinstance(x, torch.Tensor):
                images = [img for img in x]
            else:
                images = x
            
            outputs = self.forward(images)
            
            # Combine all detections into batched tensors
            # Note: Detection outputs have variable length per image
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for det in outputs:
                all_boxes.append(det["boxes"])
                all_scores.append(det["scores"])
                all_labels.append(det["labels"])
            
            # For the ModelOutput, we return the first image's outputs
            # or pad/truncate to consistent size
            return ModelOutput(
                predictions=all_labels[0] if all_labels else torch.tensor([]),
                logits=all_scores[0] if all_scores else torch.tensor([]),
                boxes=all_boxes[0] if all_boxes else torch.tensor([]),
                scores=all_scores[0] if all_scores else torch.tensor([]),
                labels=all_labels[0] if all_labels else torch.tensor([]),
            )
    
    def predict_batch(
        self, 
        images: List[torch.Tensor],
        score_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference on a batch of images and return all detections.
        
        Args:
            images: List of image tensors.
            score_threshold: Minimum score to keep detections.
            
        Returns:
            List of detection dictionaries, one per image.
        """
        with torch.no_grad():
            outputs = self.forward(images)
            
            # Filter by score threshold
            filtered_outputs = []
            for det in outputs:
                keep = det["scores"] >= score_threshold
                filtered_outputs.append({
                    "boxes": det["boxes"][keep],
                    "scores": det["scores"][keep],
                    "labels": det["labels"][keep],
                })
            
            return filtered_outputs
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._preprocessing_config


@register_model("fasterrcnn_resnet50")
class FasterRCNNResNet50(DetectionModel):
    """
    Faster R-CNN with ResNet-50 FPN backbone.
    """
    
    def __init__(
        self,
        weights: str = "COCO_V1",
        num_classes: int = 91,
        **kwargs,
    ):
        """
        Initialize Faster R-CNN ResNet-50.
        
        Args:
            weights: Pretrained weights name.
            num_classes: Number of classes (91 for COCO).
        """
        # Get weights
        if weights and weights != "None":
            weights_enum = getattr(
                detection_models.FasterRCNN_ResNet50_FPN_V2_Weights,
                weights,
                detection_models.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            )
            backbone = detection_models.fasterrcnn_resnet50_fpn_v2(weights=weights_enum)
        else:
            backbone = detection_models.fasterrcnn_resnet50_fpn_v2(weights=None)
        
        super().__init__(
            name="fasterrcnn_resnet50",
            backbone=backbone,
            num_classes=num_classes,
        )
        
        logger.info(f"Initialized Faster R-CNN ResNet-50 FPN v2 with weights: {weights}")


@register_model("retinanet_resnet50")
class RetinaNetResNet50(DetectionModel):
    """
    RetinaNet with ResNet-50 FPN backbone.
    """
    
    def __init__(
        self,
        weights: str = "COCO_V1",
        num_classes: int = 91,
        **kwargs,
    ):
        """
        Initialize RetinaNet ResNet-50.
        
        Args:
            weights: Pretrained weights name.
            num_classes: Number of classes (91 for COCO).
        """
        # Get weights
        if weights and weights != "None":
            weights_enum = getattr(
                detection_models.RetinaNet_ResNet50_FPN_V2_Weights,
                weights,
                detection_models.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            )
            backbone = detection_models.retinanet_resnet50_fpn_v2(weights=weights_enum)
        else:
            backbone = detection_models.retinanet_resnet50_fpn_v2(weights=None)
        
        super().__init__(
            name="retinanet_resnet50",
            backbone=backbone,
            num_classes=num_classes,
        )
        
        logger.info(f"Initialized RetinaNet ResNet-50 FPN v2 with weights: {weights}")


@register_model("fcos_resnet50")
class FCOSResNet50(DetectionModel):
    """
    FCOS with ResNet-50 FPN backbone.
    """
    
    def __init__(
        self,
        weights: str = "COCO_V1",
        num_classes: int = 91,
        **kwargs,
    ):
        """
        Initialize FCOS ResNet-50.
        
        Args:
            weights: Pretrained weights name.
            num_classes: Number of classes (91 for COCO).
        """
        # Get weights
        if weights and weights != "None":
            weights_enum = getattr(
                detection_models.FCOS_ResNet50_FPN_Weights,
                weights,
                detection_models.FCOS_ResNet50_FPN_Weights.DEFAULT,
            )
            backbone = detection_models.fcos_resnet50_fpn(weights=weights_enum)
        else:
            backbone = detection_models.fcos_resnet50_fpn(weights=None)
        
        super().__init__(
            name="fcos_resnet50",
            backbone=backbone,
            num_classes=num_classes,
        )
        
        logger.info(f"Initialized FCOS ResNet-50 FPN with weights: {weights}")


@register_model("ssd300_vgg16")
class SSD300VGG16(DetectionModel):
    """
    SSD300 with VGG16 backbone.
    """
    
    def __init__(
        self,
        weights: str = "COCO_V1",
        num_classes: int = 91,
        **kwargs,
    ):
        """
        Initialize SSD300 VGG16.
        
        Args:
            weights: Pretrained weights name.
            num_classes: Number of classes (91 for COCO).
        """
        # Get weights
        if weights and weights != "None":
            weights_enum = getattr(
                detection_models.SSD300_VGG16_Weights,
                weights,
                detection_models.SSD300_VGG16_Weights.DEFAULT,
            )
            backbone = detection_models.ssd300_vgg16(weights=weights_enum)
        else:
            backbone = detection_models.ssd300_vgg16(weights=None)
        
        super().__init__(
            name="ssd300_vgg16",
            backbone=backbone,
            num_classes=num_classes,
            preprocessing_config={
                "min_size": 300,
                "max_size": 300,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        )
        
        logger.info(f"Initialized SSD300 VGG16 with weights: {weights}")


# =============================================================================
# YOLO Models (using ultralytics)
# =============================================================================

class YOLOWrapper(nn.Module):
    """
    Wrapper to make YOLO model compatible with torchvision-style detection interface.
    
    This adapts the ultralytics YOLO API to return outputs in the same format
    as torchvision detection models: List[Dict[str, Tensor]] with boxes, scores, labels.
    
    IMPORTANT: We store the YOLO model as a regular attribute (not a submodule)
    to prevent PyTorch's train()/eval() from calling ultralytics' train() method
    which would start training instead of setting the mode.
    
    NOTE: Input images should be tensors in [0, 1] range (from ToTensor transform).
    YOLO handles its own internal preprocessing (resize to 640, etc.).
    Output boxes are in original image coordinates.
    """
    
    def __init__(self, yolo_model):
        super().__init__()
        # Store as regular attribute, NOT a submodule
        # This prevents train()/eval() from propagating to ultralytics
        self._yolo_model = [yolo_model]  # Wrap in list to hide from nn.Module
    
    @property
    def yolo(self):
        """Access the underlying YOLO model."""
        return self._yolo_model[0]
    
    def train(self, mode: bool = True):
        """Override train to prevent calling ultralytics train()."""
        # Only set our own training mode, don't propagate to YOLO
        self.training = mode
        return self
    
    def eval(self):
        """Override eval to prevent calling ultralytics train(False)."""
        return self.train(False)
    
    def forward(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Run YOLO inference and return torchvision-compatible outputs.
        
        Args:
            images: List of images. Can be:
                   - numpy arrays in HWC format (0-255 range)
                   - tensors in CHW format (0-1 range)
                   YOLO handles its own resizing internally.
                   Output boxes are in original image coordinates.
            
        Returns:
            List of dicts with 'boxes', 'scores', 'labels' for each image.
        """
        import numpy as np
        outputs = []
        if not isinstance(images, (list, tuple)):
            images = [images]

        processed_images = []
        for img in images:
            # Convert tensor to numpy if needed
            if isinstance(img, torch.Tensor):
                # Convert CHW tensor to HWC numpy array
                if img.dim() == 3 and img.shape[0] == 3:
                    img = img.permute(1, 2, 0)  # CHW -> HWC
                img = img.cpu().numpy()
                # Scale to 0-255 if in 0-1 range
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)

            processed_images.append(img)

        # Run one batched predict call instead of per-image calls.
        # This is much more stable for TensorRT-backed models and faster.
        results = self.yolo.predict(
            source=processed_images,  # list of numpy HWC images
            verbose=False,
            save=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show=False,
            conf=0.001,  # Low threshold, filter later if needed
            iou=0.7,
        )

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                outputs.append({
                    "boxes": boxes.xyxy.cpu(),  # (N, 4) in xyxy format
                    "scores": boxes.conf.cpu(),  # (N,) confidence scores
                    "labels": boxes.cls.cpu().long(),  # (N,) class indices
                })
            else:
                # No detections
                outputs.append({
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                })
        
        return outputs

class YOLODetectionModel(BaseModel):
    """
    YOLO detection model wrapper.
    
    Wraps ultralytics YOLO models with our BaseModel interface.
    """
    
    def __init__(
        self,
        name: str,
        model_variant: str = "yolov8n",
        weights: Optional[str] = None,
        num_classes: int = 80,  # COCO has 80 classes in YOLO
    ):
        """
        Initialize YOLO model.
        
        Args:
            name: Model name for identification.
            model_variant: YOLO variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x).
            weights: Path to custom weights or None for pretrained COCO weights.
            num_classes: Number of classes.
        """
        super().__init__(name, task="detection", num_classes=num_classes)
        
        # Disable ultralytics auto-download via environment variables BEFORE importing
        import os
        os.environ["YOLO_AUTODOWNLOAD"] = "false"
        os.environ["YOLO_OFFLINE"] = "true"
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO models. "
                "Install with: pip install ultralytics"
            )
        
        # Load YOLO model (just weights, no dataset setup)
        if weights and weights not in ["COCO", "coco", "None", None]:
            # Custom weights path
            self._yolo = YOLO(weights, task="detect")
        else:
            # Pretrained COCO weights - specify task to avoid training setup
            self._yolo = YOLO(f"{model_variant}.pt", task="detect")
        
        # Override settings to ensure inference-only mode
        self._yolo.overrides.update({
            "task": "detect",
            "mode": "predict",
            "data": None,  # Don't use any dataset yaml
            "verbose": False,
        })
        
        # Wrap for consistent interface
        self.backbone = YOLOWrapper(self._yolo)
        
        self._preprocessing_config = {
            "input_size": 640,
            "mean": [0.0, 0.0, 0.0],  # YOLO uses [0, 255] range internally
            "std": [1.0, 1.0, 1.0],
        }
        
        self._model_variant = model_variant
        logger.info(f"Initialized {model_variant.upper()} with COCO pretrained weights")
    
    def train(self, mode: bool = True):
        """
        Override train to prevent calling ultralytics train().
        
        The ultralytics YOLO model has a custom train() method that starts
        training instead of setting training mode. We override this to
        only set our own training flag.
        """
        self.training = mode
        # Set backbone's training mode (our YOLOWrapper handles this safely)
        if hasattr(self, 'backbone'):
            self.backbone.train(mode)
        return self
    
    def eval(self):
        """Override eval to prevent calling ultralytics train(False)."""
        return self.train(False)
    
    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Forward pass returning detection outputs."""
        if isinstance(x, torch.Tensor):
            images = [img for img in x]
        else:
            images = x
        return self.backbone(images)
    
    def predict(self, x: torch.Tensor) -> ModelOutput:
        """Run inference and return structured output."""
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                images = [img for img in x]
            else:
                images = x
            
            outputs = self.forward(images)
            
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for det in outputs:
                all_boxes.append(det["boxes"])
                all_scores.append(det["scores"])
                all_labels.append(det["labels"])
            
            return ModelOutput(
                predictions=all_labels[0] if all_labels else torch.tensor([]),
                logits=all_scores[0] if all_scores else torch.tensor([]),
                boxes=all_boxes[0] if all_boxes else torch.tensor([]),
                scores=all_scores[0] if all_scores else torch.tensor([]),
                labels=all_labels[0] if all_labels else torch.tensor([]),
            )
    
    def predict_batch(
        self,
        images: List[torch.Tensor],
        score_threshold: float = 0.25,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run inference on a batch with score filtering."""
        with torch.no_grad():
            outputs = self.forward(images)
            
            filtered = []
            for det in outputs:
                keep = det["scores"] >= score_threshold
                filtered.append({
                    "boxes": det["boxes"][keep],
                    "scores": det["scores"][keep],
                    "labels": det["labels"][keep],
                })
            return filtered
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._preprocessing_config
    
    def count_parameters(self) -> tuple:
        """Count parameters in the YOLO model."""
        # YOLO model parameters
        total = sum(p.numel() for p in self._yolo.model.parameters())
        trainable = sum(p.numel() for p in self._yolo.model.parameters() if p.requires_grad)
        return total, trainable


@register_model("yolov8n")
class YOLOv8Nano(YOLODetectionModel):
    """YOLOv8 Nano - smallest and fastest."""
    
    def __init__(self, weights: str = "COCO", num_classes: int = 80, **kwargs):
        super().__init__(
            name="yolov8n",
            model_variant="yolov8n",
            weights=weights,
            num_classes=num_classes,
        )


@register_model("yolov8s")
class YOLOv8Small(YOLODetectionModel):
    """YOLOv8 Small."""
    
    def __init__(self, weights: str = "COCO", num_classes: int = 80, **kwargs):
        super().__init__(
            name="yolov8s",
            model_variant="yolov8s",
            weights=weights,
            num_classes=num_classes,
        )


@register_model("yolov8m")
class YOLOv8Medium(YOLODetectionModel):
    """YOLOv8 Medium."""
    
    def __init__(self, weights: str = "COCO", num_classes: int = 80, **kwargs):
        super().__init__(
            name="yolov8m",
            model_variant="yolov8m",
            weights=weights,
            num_classes=num_classes,
        )


@register_model("yolov8l")
class YOLOv8Large(YOLODetectionModel):
    """YOLOv8 Large."""
    
    def __init__(self, weights: str = "COCO", num_classes: int = 80, **kwargs):
        super().__init__(
            name="yolov8l",
            model_variant="yolov8l",
            weights=weights,
            num_classes=num_classes,
        )


@register_model("yolov8x")
class YOLOv8XLarge(YOLODetectionModel):
    """YOLOv8 Extra Large - largest and most accurate."""
    
    def __init__(self, weights: str = "COCO", num_classes: int = 80, **kwargs):
        super().__init__(
            name="yolov8x",
            model_variant="yolov8x",
            weights=weights,
            num_classes=num_classes,
        )
