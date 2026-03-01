"""
COCO dataset loader.

This module provides data loading for COCO 2017 dataset.
Supports both classification (using image-level labels) and
object detection tasks.

Expected directory structure:
    coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   └── ...
    ├── train2017/
    │   └── *.jpg
    └── val2017/
        └── *.jpg
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.datasets.base import (
    BaseDataset,
    SubsetDataset,
    get_preprocessing_transform,
)
from src.utils.config import DatasetConfig
from src.utils.logging import get_logger

logger = get_logger("qda.datasets.coco")


# COCO category information
COCO_NUM_CLASSES = 91  # Including background


class COCODataset(BaseDataset):
    """
    COCO dataset wrapper for classification and detection.
    
    For classification: Returns image and list of category IDs present.
    For detection: Returns image and target dict with boxes, labels, etc.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "val2017",
        task: str = "classification",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize COCO dataset.
        
        Args:
            root: Root directory containing annotations/ and images.
            split: Dataset split ('train2017' or 'val2017').
            task: Task type ('classification' or 'detection').
            transform: Optional transform for images.
            target_transform: Optional transform for targets.
        """
        super().__init__(root, transform, target_transform)
        
        self.split = split
        self.task = task
        self.images_dir = Path(root) / split
        self.annotations_file = Path(root) / "annotations" / f"instances_{split}.json"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"COCO images directory not found: {self.images_dir}. "
                f"Run scripts/download_coco.sh to download the dataset."
            )
        
        if not self.annotations_file.exists():
            raise FileNotFoundError(
                f"COCO annotations not found: {self.annotations_file}. "
                f"Run scripts/download_coco.sh to download the dataset."
            )
        
        # Load COCO annotations
        self._load_annotations()
        
        logger.info(
            f"Loaded COCO {split}: {len(self.image_ids)} images, "
            f"task={task}"
        )
    
    def _load_annotations(self) -> None:
        """Load COCO annotations using pycocotools."""
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError(
                "pycocotools is required for COCO dataset. "
                "Install with: pip install pycocotools"
            )
        
        self.coco = COCO(str(self.annotations_file))
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        
        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_id_to_name = {
            cat["id"]: cat["name"] for cat in self.categories
        }
        
        # Create contiguous category mapping (COCO IDs are not contiguous)
        self.category_id_to_idx = {
            cat_id: idx for idx, cat_id in enumerate(sorted(self.coco.getCatIds()))
        }
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a sample.
        
        For classification: Returns (image, multi-hot label vector).
        For detection: Returns (image, target_dict).
        """
        image_id = self.image_ids[index]
        
        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        if self.task == "classification":
            target = self._get_classification_target(annotations)
        else:  # detection
            target = self._get_detection_target(annotations, image, image_id)
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            # For detection without transform, convert PIL to numpy array
            # This format works with both YOLO (expects numpy HWC 0-255)
            # and torchvision detection models (we'll convert to tensor later)
            import numpy as np
            image = np.array(image)  # HWC format, 0-255 range
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def _get_classification_target(
        self, 
        annotations: List[Dict],
    ) -> torch.Tensor:
        """
        Get multi-hot classification target.
        
        Returns:
            Tensor of shape (num_classes,) with 1s for present categories.
        """
        target = torch.zeros(len(self.category_id_to_idx))
        
        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id in self.category_id_to_idx:
                idx = self.category_id_to_idx[cat_id]
                target[idx] = 1
        
        return target
    
    def _get_detection_target(
        self,
        annotations: List[Dict],
        image: Image.Image,
        image_id: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Get detection target with boxes and labels.
        
        Returns:
            Dictionary with boxes, labels, area, iscrowd, image_id.
        """
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            if ann.get("iscrowd", 0):
                continue  # Skip crowd annotations for training
            
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            
            # Keep original COCO category ID for proper COCO evaluation
            # (COCOeval expects original COCO category IDs, not contiguous indices)
            cat_id = ann["category_id"]
            labels.append(cat_id)  # Use original COCO category ID
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))
        
        # Handle images with no annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        return {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
            "image_id": image_id,
        }
    
    @property
    def num_classes(self) -> int:
        return len(self.category_id_to_idx)
    
    @property
    def class_names(self) -> List[str]:
        return [
            self.category_id_to_name[cat_id]
            for cat_id in sorted(self.category_id_to_idx.keys())
        ]


def get_coco_loader(
    config: DatasetConfig,
    task: str = "classification",
    model_name: str = "resnet50",
    num_workers: Optional[int] = None,
    debug: bool = False,
    debug_samples: int = 100,
) -> DataLoader:
    """
    Create a DataLoader for COCO.
    
    Args:
        config: Dataset configuration.
        task: Task type ('classification' or 'detection').
        model_name: Model name for preprocessing.
        num_workers: Number of data loading workers.
        debug: Whether to use a small subset.
        debug_samples: Number of samples in debug mode.
        
    Returns:
        DataLoader for COCO.
    """
    # For detection, use minimal transforms to preserve original coordinates
    # Detection models (YOLO, FasterRCNN) handle their own preprocessing
    if task == "detection":
        # No transform - return numpy array for YOLO compatibility
        # YOLO expects numpy arrays in HWC format with values 0-255
        # or PIL images, which it handles internally
        transform = None
    else:
        transform = get_preprocessing_transform(model_name, is_training=False)
    
    split = config.split or "val2017"
    dataset = COCODataset(
        root=config.root,
        split=split,
        task=task,
        transform=transform,
    )
    
    if debug:
        dataset = SubsetDataset(dataset, debug_samples)
        logger.info(f"Debug mode: using {debug_samples} samples")
    
    # Detection requires special collate function
    collate_fn = None
    if task == "detection":
        collate_fn = detection_collate_fn
    
    worker_count = config.num_workers if num_workers is None else int(num_workers)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=worker_count,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    logger.info(
        f"Created COCO loader ({task}): {len(dataset)} samples, "
        f"batch_size={config.batch_size}"
    )
    
    return loader


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for detection that handles variable-size targets.
    
    Args:
        batch: List of (image, target) tuples.
        
    Returns:
        Tuple of (list of images, list of targets).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
