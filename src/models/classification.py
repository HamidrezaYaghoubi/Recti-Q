"""
Classification models for Recti-Q (IROS 2026).

Provides timm-based classification via TimmClassifier and the
forward_features_logits() helper the Recti-Q adapter module uses.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import timm
import timm.data

from src.models.base import BaseModel, ModelOutput
from src.models.factory import register_model
from src.utils.logging import get_logger

logger = get_logger("recti_q.models.classification")

# Map short alias -> timm model name
_TIMM_ARCH_MAP = {
    "resnet50":              "resnet50",
    "deit_tiny":             "deit_tiny_patch16_224",
    "deit_small":            "deit_small_patch16_224",
    "deit_base":             "deit_base_patch16_224",
    "deit_tiny_patch16_224":  "deit_tiny_patch16_224",
    "deit_small_patch16_224": "deit_small_patch16_224",
    "deit_base_patch16_224":  "deit_base_patch16_224",
}


def forward_features_logits(backbone: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (u, z) from a single forward pass through a timm backbone.

    u : pre-classifier feature vector  (B, d)
    z : logits                          (B, C)

    Works on any timm model (fp32 or quantized copy).
    """
    feats = backbone.forward_features(x)
    u = backbone.forward_head(feats, pre_logits=True)
    z = backbone.forward_head(feats)
    return u, z


class ClassificationModel(BaseModel):
    """Base class for classification models.

    Holds self.backbone and provides forward / predict / get_preprocessing_config.
    """

    def __init__(self, name: str, backbone: nn.Module, num_classes: int = 1000):
        super().__init__(name, task="classification", num_classes=num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> ModelOutput:
        """Run inference and return structured ModelOutput."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)
            return ModelOutput(
                predictions=predictions,
                logits=logits,
                probabilities=probs,
                confidences=confidences,
            )

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Return preprocessing config dict (input_size, mean, std)."""
        raise NotImplementedError("Subclasses must implement get_preprocessing_config")


class TimmClassifier(ClassificationModel):
    """Wraps a timm pretrained model for Recti-Q classification.

    Args:
        name:          Logical model name (e.g. "resnet50").
        architecture:  timm model name or short alias (see _TIMM_ARCH_MAP).
        weights:       "pretrained" -> pretrained=True; "none"/"scratch"/"" -> False.
        num_classes:   Number of output classes (passed to timm.create_model).
    """

    def __init__(
        self,
        name: str,
        architecture: str,
        weights: str = "pretrained",
        num_classes: int = 1000,
    ):
        timm_name = _TIMM_ARCH_MAP.get(architecture.lower(), architecture)
        pretrained = weights.lower() not in {"none", "scratch", ""}

        backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        super().__init__(name=name, backbone=backbone, num_classes=num_classes)

        # Cache data config once
        self._data_cfg = timm.data.resolve_model_data_config(self.backbone)

        logger.info(f"TimmClassifier: {timm_name}, pretrained={pretrained}, num_classes={num_classes}")

    @property
    def classifier_dims(self) -> Tuple[int, int]:
        """Return (d, C): in_features and out_features of the classifier head."""
        clf = self.backbone.get_classifier()
        if hasattr(clf, "in_features") and hasattr(clf, "out_features"):
            return clf.in_features, clf.out_features
        # Fallback: timm exposes num_features for the backbone output dim
        return self.backbone.num_features, self._num_classes

    def build_transform(self, train: bool = False):
        """Return the timm eval or train transform for this model."""
        return timm.data.create_transform(**self._data_cfg, is_training=train)

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Return dict with input_size, mean, std from the timm data config."""
        cfg = self._data_cfg
        return {
            "input_size": cfg.get("input_size", (3, 224, 224)),
            "mean": list(cfg.get("mean", (0.485, 0.456, 0.406))),
            "std": list(cfg.get("std", (0.229, 0.224, 0.225))),
        }


# ── Register the four paper models ──
# ModelFactory.create() will call the builder registered here.

def _timm_builder(name: str, architecture: str):
    """Return a builder that ModelFactory can call with (weights, num_classes)."""
    def builder(weights: str = "pretrained", num_classes: int = 1000) -> TimmClassifier:
        return TimmClassifier(name=name, architecture=architecture, weights=weights, num_classes=num_classes)
    return builder


for _short, _timm in [
    ("resnet50",    "resnet50"),
    ("deit_tiny",   "deit_tiny_patch16_224"),
    ("deit_small",  "deit_small_patch16_224"),
    ("deit_base",   "deit_base_patch16_224"),
]:
    register_model(_short)(_timm_builder(_short, _timm))
    # Also register the full timm name as an alias
    if _short != _timm:
        register_model(_timm)(_timm_builder(_short, _timm))
