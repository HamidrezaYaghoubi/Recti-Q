"""
BDD100K YOLO-format dataset loader for detection.

Expected layout:
  root/
    data.yaml (optional)
    train/images, train/labels
    val/images, val/labels
    test/images, test/labels
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.base import BaseDataset, SubsetDataset
from src.utils.config import DatasetConfig
from src.utils.logging import get_logger

logger = get_logger("qda.datasets.bdd100k")


DEFAULT_BDD100K_CLASS_NAMES = [
    "person",
    "rider",
    "car",
    "bus",
    "truck",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
]


class BDD100KDetectionDataset(BaseDataset):
    """BDD100K dataset wrapper for YOLO-format detection labels."""

    def __init__(
        self,
        root: str,
        split: str = "val",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, target_transform)
        self.split = split
        self.images_dir = Path(root) / split / "images"
        self.labels_dir = Path(root) / split / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"BDD100K images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"BDD100K labels directory not found: {self.labels_dir}")

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self._class_names = self._load_class_names()
        logger.info(f"Loaded BDD100K {split}: {len(self.image_paths)} images")

    def _load_class_names(self) -> List[str]:
        data_yaml = Path(self.root) / "data.yaml"
        if not data_yaml.exists():
            return DEFAULT_BDD100K_CLASS_NAMES
        try:
            import yaml

            with open(data_yaml, "r") as f:
                payload = yaml.safe_load(f) or {}
            names = payload.get("names")
            if isinstance(names, list) and names:
                return [str(x) for x in names]
            if isinstance(names, dict) and names:
                return [str(names[k]) for k in sorted(names.keys(), key=lambda v: int(v))]
        except Exception:
            pass
        return DEFAULT_BDD100K_CLASS_NAMES

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        target = self._load_target(label_path=label_path, width=width, height=height, image_id=index)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = np.array(image)  # HWC, uint8

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def _load_target(
        self,
        label_path: Path,
        width: int,
        height: int,
        image_id: int,
    ) -> Dict[str, torch.Tensor]:
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = map(float, parts)
                    x1 = (cx - bw / 2.0) * width
                    y1 = (cy - bh / 2.0) * height
                    x2 = (cx + bw / 2.0) * width
                    y2 = (cy + bh / 2.0) * height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls_id))
                    areas.append(max((x2 - x1), 0.0) * max((y2 - y1), 0.0))

        if not boxes:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor(image_id, dtype=torch.int64),
            }

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
        }

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_names(self) -> List[str]:
        return self._class_names


def bdd_detection_collate_fn(
    batch: List[Tuple[Any, Dict[str, torch.Tensor]]]
) -> Tuple[List[Any], List[Dict[str, torch.Tensor]]]:
    images = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    return images, targets


def get_bdd100k_loader(
    config: DatasetConfig,
    task: str = "detection",
    num_workers: Optional[int] = None,
    debug: bool = False,
    debug_samples: int = 100,
) -> DataLoader:
    """
    Create BDD100K loader.

    Only detection task is supported.
    """
    if task != "detection":
        raise ValueError("BDD100K loader currently supports task='detection' only.")

    split = config.split or "val"
    dataset = BDD100KDetectionDataset(
        root=config.root,
        split=split,
        transform=None,  # keep raw image for YOLO/torchvision compatibility
    )

    if debug:
        dataset = SubsetDataset(dataset, debug_samples)
        logger.info(f"Debug mode: using {debug_samples} samples")

    worker_count = config.num_workers if num_workers is None else int(num_workers)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=worker_count,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=bdd_detection_collate_fn,
    )

    logger.info(
        f"Created BDD100K loader (detection): {len(dataset)} samples, batch_size={config.batch_size}"
    )
    return loader
