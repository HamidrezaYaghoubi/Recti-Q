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
    transform: Optional[Callable] = None,
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
    if transform is None:
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
    worker_count = config.num_workers if num_workers is None else int(num_workers)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=worker_count,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    
    logger.info(
        f"Created ImageNet loader: {len(dataset)} samples, "
        f"batch_size={config.batch_size}, num_workers={loader.num_workers}"
    )
    
    return loader


def get_imagenet_subset_loader(
    config: DatasetConfig,
    model_name: str = "resnet50",
    transform: Optional[Callable] = None,
    num_workers: Optional[int] = None,
    debug: bool = False,
    debug_samples: int = 100,
) -> DataLoader:
    """
    Create a DataLoader for a class-balanced subset of ImageNet TRAIN.

    Selects `config.subset_fraction` (default 0.05) of each class's images
    using `config.subset_seed` for reproducibility, giving a 5% balanced
    subsample (~64k images) suitable for Recti-Q adapter training.

    Args:
        config: Dataset configuration. Uses config.train_root if set; otherwise
                derives <parent-of-config.root>/train (handles root pointing at
                .../imagenet/validation).
        model_name: Model name for default preprocessing transform.
        transform: Override transform. If None, uses training augmentations.
        num_workers: Override worker count; falls back to config.num_workers.
        debug: Cap dataset to debug_samples for fast iteration.
        debug_samples: Number of samples in debug mode.

    Returns:
        DataLoader with shuffled class-balanced subset.
    """
    import random
    from collections import defaultdict
    from torch.utils.data import Subset

    # Resolve train root
    if config.train_root:
        train_root = Path(config.train_root)
    else:
        # config.root may point at .../imagenet/validation; go up one and find train
        root_path = Path(config.root)
        if root_path.name in ("validation", "val", "train"):
            train_root = root_path.parent / "train"
        else:
            train_root = root_path / "train"

    if not train_root.exists():
        raise FileNotFoundError(f"ImageNet train directory not found: {train_root}")

    # Load full train set (no transform yet — we apply after subset selection)
    full_dataset = ImageFolder(str(train_root))

    # Class-balanced sampling
    fraction = config.subset_fraction if config.subset_fraction else 0.05
    seed = config.subset_seed if config.subset_seed is not None else 42

    # Group indices by class
    class_to_indices: dict = defaultdict(list)
    for idx, label in enumerate(full_dataset.targets):
        class_to_indices[label].append(idx)

    rng = random.Random(seed)
    selected: List[int] = []
    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        k = max(1, round(fraction * len(indices)))
        chosen = rng.sample(indices, min(k, len(indices)))
        selected.extend(chosen)

    selected.sort()  # deterministic order before shuffle in DataLoader

    # Apply transform
    if transform is None:
        transform = get_preprocessing_transform(model_name, is_training=True)
    full_dataset.transform = transform

    subset = Subset(full_dataset, selected)

    # Debug cap
    if debug:
        subset = SubsetDataset(subset, debug_samples, shuffle=True, seed=seed)
        logger.info(f"Debug mode: using {len(subset)} samples")

    worker_count = config.num_workers if num_workers is None else int(num_workers)
    loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    logger.info(
        f"Created ImageNet subset loader: {len(subset)} samples "
        f"({fraction*100:.1f}% balanced), batch_size={config.batch_size}"
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
