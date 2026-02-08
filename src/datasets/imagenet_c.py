"""
ImageNet-C dataset loader.

This module provides data loading for ImageNet-C (corrupted ImageNet).

ImageNet-C contains 15 corruption types at 5 severity levels:
- Noise: gaussian_noise, shot_noise, impulse_noise
- Blur: defocus_blur, glass_blur, motion_blur, zoom_blur
- Weather: snow, frost, fog, brightness
- Digital: contrast, elastic_transform, pixelate, jpeg_compression

Expected directory structure:
    imagenet-c/
    ├── gaussian_noise/
    │   ├── 1/  # severity level
    │   │   ├── n01440764/
    │   │   │   └── ILSVRC2012_val_00000293.JPEG
    │   │   └── ...
    │   ├── 2/
    │   ├── 3/
    │   ├── 4/
    │   └── 5/
    ├── shot_noise/
    └── ...
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

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

logger = get_logger("qda.datasets.imagenet_c")


# All corruption types in ImageNet-C
CORRUPTION_TYPES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

ALL_CORRUPTIONS = [c for group in CORRUPTION_TYPES.values() for c in group]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]


class ImageNetCDataset(BaseDataset):
    """
    ImageNet-C dataset for a specific corruption and severity.
    """
    
    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize ImageNet-C dataset.
        
        Args:
            root: Root directory of ImageNet-C.
            corruption: Corruption type (e.g., 'gaussian_noise').
            severity: Severity level (1-5).
            transform: Optional transform for images.
            target_transform: Optional transform for labels.
        """
        super().__init__(root, transform, target_transform)
        
        if corruption not in ALL_CORRUPTIONS:
            raise ValueError(
                f"Unknown corruption: {corruption}. "
                f"Available: {ALL_CORRUPTIONS}"
            )
        
        if severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"Invalid severity: {severity}. Must be in {SEVERITY_LEVELS}"
            )
        
        self.corruption = corruption
        self.severity = severity
        self.data_dir = Path(root) / corruption / str(severity)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"ImageNet-C directory not found: {self.data_dir}"
            )
        
        self._dataset = ImageFolder(
            str(self.data_dir),
            transform=transform,
            target_transform=target_transform,
        )
        
        logger.info(
            f"Loaded ImageNet-C {corruption} (severity {severity}): "
            f"{len(self._dataset)} images"
        )
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self._dataset[index]
    
    @property
    def num_classes(self) -> int:
        return 1000
    
    @property
    def class_names(self) -> List[str]:
        return self._dataset.classes
    
    @property
    def samples(self) -> List[Tuple[str, int]]:
        return self._dataset.samples


class ImageNetCMultiCorruption:
    """
    Iterator over multiple ImageNet-C corruptions and severities.
    
    Useful for evaluating robustness across all corruption types.
    """
    
    def __init__(
        self,
        root: str,
        corruptions: Optional[List[str]] = None,
        severities: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize multi-corruption iterator.
        
        Args:
            root: Root directory of ImageNet-C.
            corruptions: List of corruption types (default: all).
            severities: List of severity levels (default: all).
            transform: Transform to apply to images.
        """
        self.root = root
        self.corruptions = corruptions or ALL_CORRUPTIONS
        self.severities = severities or SEVERITY_LEVELS
        self.transform = transform
        
        # Validate corruptions
        for c in self.corruptions:
            if c not in ALL_CORRUPTIONS:
                raise ValueError(f"Unknown corruption: {c}")
    
    def __iter__(self) -> Iterator[Tuple[str, int, ImageNetCDataset]]:
        """
        Iterate over all corruption/severity combinations.
        
        Yields:
            Tuples of (corruption_name, severity, dataset).
        """
        for corruption in self.corruptions:
            for severity in self.severities:
                try:
                    dataset = ImageNetCDataset(
                        root=self.root,
                        corruption=corruption,
                        severity=severity,
                        transform=self.transform,
                    )
                    yield corruption, severity, dataset
                except FileNotFoundError as e:
                    logger.warning(f"Skipping {corruption}/{severity}: {e}")
                    continue
    
    def get_all_loaders(
        self,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> Dict[Tuple[str, int], DataLoader]:
        """
        Get DataLoaders for all corruption/severity combinations.
        
        Returns:
            Dictionary mapping (corruption, severity) to DataLoader.
        """
        loaders = {}
        for corruption, severity, dataset in self:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            loaders[(corruption, severity)] = loader
        return loaders


def get_imagenet_c_loader(
    config: DatasetConfig,
    corruption: str,
    severity: int,
    model_name: str = "resnet50",
    num_workers: Optional[int] = None,
    debug: bool = False,
    debug_samples: int = 100,
) -> DataLoader:
    """
    Create a DataLoader for a specific ImageNet-C corruption.
    
    Args:
        config: Dataset configuration.
        corruption: Corruption type.
        severity: Severity level (1-5).
        model_name: Model name for preprocessing.
        num_workers: Number of data loading workers.
        debug: Whether to use a small subset.
        debug_samples: Number of samples in debug mode.
        
    Returns:
        DataLoader for ImageNet-C.
    """
    transform = get_preprocessing_transform(model_name, is_training=False)
    
    dataset = ImageNetCDataset(
        root=config.root,
        corruption=corruption,
        severity=severity,
        transform=transform,
    )
    
    if debug:
        dataset = SubsetDataset(dataset, debug_samples)
        logger.info(f"Debug mode: using {debug_samples} samples")
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=num_workers or config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    
    return loader


def get_all_imagenet_c_loaders(
    config: DatasetConfig,
    model_name: str = "resnet50",
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    num_workers: Optional[int] = None,
) -> Dict[Tuple[str, int], DataLoader]:
    """
    Create DataLoaders for all specified ImageNet-C corruptions.
    
    Args:
        config: Dataset configuration.
        model_name: Model name for preprocessing.
        corruptions: List of corruption types (uses config if None).
        severities: List of severity levels (uses config if None).
        num_workers: Number of data loading workers.
        
    Returns:
        Dictionary mapping (corruption, severity) to DataLoader.
    """
    corruptions = corruptions or config.corruptions or ALL_CORRUPTIONS
    severities = severities or config.severities or SEVERITY_LEVELS
    
    transform = get_preprocessing_transform(model_name, is_training=False)
    
    multi_dataset = ImageNetCMultiCorruption(
        root=config.root,
        corruptions=corruptions,
        severities=severities,
        transform=transform,
    )
    
    return multi_dataset.get_all_loaders(
        batch_size=config.batch_size,
        num_workers=num_workers or config.num_workers,
        pin_memory=config.pin_memory,
    )


def get_corruption_group(corruption: str) -> str:
    """
    Get the corruption group (noise, blur, weather, digital) for a corruption type.
    
    Args:
        corruption: Corruption type name.
        
    Returns:
        Corruption group name.
    """
    for group, corruptions in CORRUPTION_TYPES.items():
        if corruption in corruptions:
            return group
    raise ValueError(f"Unknown corruption: {corruption}")
