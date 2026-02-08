"""
ImageNet dataset loader.

This module provides data loading for ImageNet-1K validation set.

Expected directory structure:
    imagenet/
    ├── train/
    │   ├── n01440764/
    │   │   ├── n01440764_10026.JPEG
    │   │   └── ...
    │   └── ...
    └── val/
        ├── n01440764/
        │   ├── ILSVRC2012_val_00000293.JPEG
        │   └── ...
        └── ...
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.datasets.base import (
    BaseDataset,
    SubsetDataset,
    get_preprocessing_transform,
)
from src.utils.config import DatasetConfig
from src.utils.logging import get_logger

logger = get_logger("qda.datasets.imagenet")


# ImageNet class names (synset IDs to human-readable names)
# This is a subset; full mapping can be loaded from imagenet_class_index.json
IMAGENET_CLASS_COUNT = 1000


class ImageNetDataset(BaseDataset):
    """
    ImageNet dataset wrapper.
    
    Wraps torchvision's ImageFolder for ImageNet data.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "val",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize ImageNet dataset.
        
        Args:
            root: Root directory containing train/ and val/ subdirectories.
            split: Dataset split ('train' or 'val').
            transform: Optional transform for images.
            target_transform: Optional transform for labels.
        """
        super().__init__(root, transform, target_transform)
        
        self.split = split
        self.data_dir = Path(root) / split
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"ImageNet {split} directory not found: {self.data_dir}"
            )
        
        # Use torchvision's ImageFolder
        self._dataset = ImageFolder(
            str(self.data_dir),
            transform=transform,
            target_transform=target_transform,
        )
        
        logger.info(
            f"Loaded ImageNet {split}: {len(self._dataset)} images, "
            f"{len(self._dataset.classes)} classes"
        )
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self._dataset[index]
    
    @property
    def num_classes(self) -> int:
        return IMAGENET_CLASS_COUNT
    
    @property
    def class_names(self) -> List[str]:
        return self._dataset.classes
    
    @property
    def samples(self) -> List[Tuple[str, int]]:
        """Get list of (image_path, class_index) tuples."""
        return self._dataset.samples
    
    @property
    def class_to_idx(self) -> dict:
        """Get mapping from class name to index."""
        return self._dataset.class_to_idx


def get_imagenet_loader(
    config: DatasetConfig,
    model_name: str = "resnet50",
    num_workers: Optional[int] = None,
    debug: bool = False,
    debug_samples: int = 100,
) -> DataLoader:
    """
    Create a DataLoader for ImageNet.
    
    Args:
        config: Dataset configuration.
        model_name: Model name for preprocessing transform.
        num_workers: Number of data loading workers.
        debug: Whether to use a small subset.
        debug_samples: Number of samples in debug mode.
        
    Returns:
        DataLoader for ImageNet.
    """
    # Get the appropriate transform
    transform = get_preprocessing_transform(model_name, is_training=False)
    
    # Create dataset
    split = config.split or "val"
    dataset = ImageNetDataset(
        root=config.root,
        split=split,
        transform=transform,
    )
    
    # Use subset for debugging
    if debug:
        dataset = SubsetDataset(dataset, debug_samples)
        logger.info(f"Debug mode: using {debug_samples} samples")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=num_workers or config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    
    logger.info(
        f"Created ImageNet loader: {len(dataset)} samples, "
        f"batch_size={config.batch_size}, num_workers={loader.num_workers}"
    )
    
    return loader


def get_imagenet_class_names() -> List[str]:
    """
    Get human-readable ImageNet class names.
    
    Returns:
        List of 1000 class names.
    """
    try:
        # Try to load from torchvision's bundled class names
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        return weights.meta["categories"]
    except Exception:
        # Return placeholder names
        return [f"class_{i}" for i in range(IMAGENET_CLASS_COUNT)]


def load_imagenet_labels(labels_path: str) -> dict:
    """
    Load ImageNet labels from a JSON file.
    
    Expected format: {"0": ["n01440764", "tench"], ...}
    
    Args:
        labels_path: Path to imagenet_class_index.json.
        
    Returns:
        Dictionary mapping class index to (synset_id, class_name).
    """
    import json
    
    with open(labels_path, "r") as f:
        labels = json.load(f)
    
    return {int(k): v for k, v in labels.items()}
