"""
Base dataset utilities and preprocessing transforms.

This module provides common functionality for all datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.
    
    Provides a consistent interface for dataset implementations.
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the base dataset.
        
        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to images.
            target_transform: Optional transform to apply to targets.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return a sample and its target."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        pass
    
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Return the list of class names."""
        pass


def get_preprocessing_transform(
    model_name: str,
    input_size: int = 224,
    is_training: bool = False,
) -> transforms.Compose:
    """
    Get the appropriate preprocessing transform for a model.
    
    Args:
        model_name: Name of the model (resnet50, mobilenetv2, vit_base, etc.).
        input_size: Input image size (default 224).
        is_training: Whether to include training augmentations.
        
    Returns:
        Composed transform pipeline.
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    # ViT models may use different input sizes
    if "vit" in model_name.lower():
        input_size = 224  # ViT-B/16 uses 224x224
    
    if is_training:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Validation/test transforms
        # Standard ImageNet preprocessing: resize to 256, center crop to 224
        resize_size = int(input_size * 256 / 224)
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    
    return transform


def get_model_specific_transform(
    architecture: str,
    weights: str,
) -> Optional[transforms.Compose]:
    """
    Get the transform associated with pretrained model weights.
    
    This uses the transforms bundled with torchvision weights
    to ensure exact preprocessing match.
    
    Args:
        architecture: Model architecture name.
        weights: Weights name (e.g., "IMAGENET1K_V2").
        
    Returns:
        Transform from the weights, or None if not available.
    """
    try:
        from torchvision.models import get_weight
        weights_obj = get_weight(f"{architecture}_{weights}")
        return weights_obj.transforms()
    except Exception:
        return None


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset and returns a subset of samples.
    
    Useful for debugging with a small number of samples.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_samples: int,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the subset dataset.
        
        Args:
            dataset: Original dataset to wrap.
            num_samples: Number of samples to include.
            shuffle: Whether to shuffle indices before selecting.
            seed: Random seed for shuffling.
        """
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        
        if shuffle:
            import numpy as np
            rng = np.random.RandomState(seed)
            self.indices = rng.permutation(len(dataset))[:self.num_samples]
        else:
            self.indices = list(range(self.num_samples))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.dataset[self.indices[index]]


class DatasetWithMetadata(Dataset):
    """
    A dataset wrapper that includes metadata (image paths, indices) with each sample.
    
    Useful for tracking which images were correctly/incorrectly classified.
    """
    
    def __init__(self, dataset: Dataset):
        """
        Initialize the wrapper.
        
        Args:
            dataset: Dataset to wrap.
        """
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample with metadata.
        
        Returns:
            Dictionary containing:
                - image: The image tensor
                - target: The target label
                - index: The sample index
                - path: Image file path (if available)
        """
        image, target = self.dataset[index]
        
        result = {
            "image": image,
            "target": target,
            "index": index,
        }
        
        # Try to get the image path
        if hasattr(self.dataset, "samples"):
            result["path"] = self.dataset.samples[index][0]
        elif hasattr(self.dataset, "imgs"):
            result["path"] = self.dataset.imgs[index][0]
        
        return result
