"""
Base model interface for all models.

This module defines the abstract base class and common interfaces
that all models must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """
    Standardized output format for classification models.

    Attributes:
        predictions: Predicted class indices.
        logits: Raw model outputs before softmax.
        probabilities: Softmax probabilities.
        confidences: Maximum probability for each prediction.
        features: Optional pre-classifier features for analysis.
    """
    predictions: torch.Tensor
    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    confidences: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key in ["predictions", "logits", "probabilities", "confidences", "features"]:
            value = getattr(self, key)
            if value is not None:
                result[key] = value.cpu() if isinstance(value, torch.Tensor) else value
        return result


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all models.
    
    Provides a consistent interface for different model architectures.
    """
    
    def __init__(
        self,
        name: str,
        task: str = "classification",
        num_classes: int = 1000,
    ):
        """
        Initialize the base model.
        
        Args:
            name: Model name/identifier.
            task: Task type ('classification' or 'detection').
            num_classes: Number of output classes.
        """
        super().__init__()
        self._name = name
        self._task = task
        self._num_classes = num_classes
        self._precision = "fp32"
    
    @property
    def name(self) -> str:
        """Get the model name."""
        return self._name
    
    @property
    def task(self) -> str:
        """Get the task type."""
        return self._task
    
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self._num_classes
    
    @property
    def precision(self) -> str:
        """Get the current precision (fp32, int8, etc.)."""
        return self._precision
    
    @precision.setter
    def precision(self, value: str) -> None:
        """Set the precision level."""
        self._precision = value
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor.
            
        Returns:
            Model output (logits for classification).
        """
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> ModelOutput:
        """
        Run inference and return structured output.
        
        Args:
            x: Input tensor.
            
        Returns:
            ModelOutput with predictions and metadata.
        """
        pass
    
    @abstractmethod
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get the preprocessing configuration for this model.
        
        Returns:
            Dictionary with preprocessing parameters.
        """
        pass
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.
        
        Returns:
            Tuple of (total_params, trainable_params).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information dictionary.
        
        Returns:
            Dictionary with model metadata.
        """
        total, trainable = self.count_parameters()
        return {
            "name": self.name,
            "task": self.task,
            "num_classes": self.num_classes,
            "precision": self.precision,
            "total_parameters": total,
            "trainable_parameters": trainable,
        }
    
    def set_eval_mode(self) -> "BaseModel":
        """Set model to evaluation mode and disable gradients."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self


class ModelWrapper(BaseModel):
    """
    Wrapper class for pre-existing PyTorch models.
    
    Adapts any nn.Module to the BaseModel interface.
    """
    
    def __init__(
        self,
        model: nn.Module,
        name: str,
        task: str = "classification",
        num_classes: int = 1000,
        preprocessing_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the wrapper.
        
        Args:
            model: PyTorch model to wrap.
            name: Model name.
            task: Task type.
            num_classes: Number of classes.
            preprocessing_config: Preprocessing configuration.
        """
        super().__init__(name, task, num_classes)
        self.model = model
        self._preprocessing_config = preprocessing_config or {
            "input_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through wrapped model."""
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> ModelOutput:
        """Run inference with wrapped model."""
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
