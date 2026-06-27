"""
PACS dataset loader for leave-one-domain-out (LODO) experiments.

Domains: photo, art_painting, cartoon, sketch.
7 classes: dog, elephant, giraffe, guitar, horse, house, person.

Split files live at:
    <pacs_root>/pacs_label/<domain>_{train,crossval,test}_kfold.txt

Each line: "<domain>/<class>/pic_XXX.jpg <label>"  (label is 1-indexed).

Images live at:
    <pacs_root>/pacs_data/pacs_data/<domain>/<class>/pic_XXX.jpg
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

from src.datasets.base import BaseDataset, get_preprocessing_transform
from src.utils.config import DatasetConfig
from src.utils.logging import get_logger

logger = get_logger("qda.datasets.pacs")

PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


class PACSDataset(BaseDataset):
    """
    PACS dataset built from one or more *_kfold.txt split files.

    Args:
        samples: List of (image_abs_path, label_0indexed) tuples.
        transform: Transform to apply to PIL images.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
    ):
        # BaseDataset expects a root; pass empty string — we use abs paths.
        super().__init__(root="", transform=transform)
        self._samples = samples

    # ------------------------------------------------------------------
    # BaseDataset abstract interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        path, label = self._samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self) -> int:
        return len(PACS_CLASSES)

    @property
    def class_names(self) -> List[str]:
        return PACS_CLASSES


# ---------------------------------------------------------------------------
# Internal helper: parse a single split file
# ---------------------------------------------------------------------------

def _parse_split_file(
    label_dir: Path,
    img_root: Path,
    domain: str,
    split: str,
) -> List[Tuple[str, int]]:
    """
    Parse <label_dir>/<domain>_<split>_kfold.txt.

    Returns list of (abs_image_path, 0-indexed label). Skips missing files
    gracefully with a warning.
    """
    txt = label_dir / f"{domain}_{split}_kfold.txt"
    if not txt.exists():
        logger.warning(f"Split file not found, skipping: {txt}")
        return []

    samples = []
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            rel_path, label_str = parts
            label_0 = int(label_str) - 1  # 1-indexed → 0-indexed
            abs_path = str(img_root / rel_path)
            samples.append((abs_path, label_0))

    return samples


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def get_pacs_loaders(
    config: DatasetConfig,
    target_domain: str,
    model_name: str = "resnet50",
    transform: Optional[Callable] = None,
    train_transform: Optional[Callable] = None,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for a PACS leave-one-domain-out experiment.

    Args:
        config: DatasetConfig with root pointing at the PACS root directory
                (e.g. .../datasets/pacs).
        target_domain: The held-out domain used for evaluation only.
        model_name: Used to build default transforms when none are provided.
        transform: Eval transform for source_val and target_test. Defaults to
                   get_preprocessing_transform(model_name, is_training=False).
        train_transform: Train transform for source_train. Defaults to
                         get_preprocessing_transform(model_name, is_training=True).

    Returns:
        Dict with keys:
            "source_train"  - DataLoader over train splits of the 3 source domains
            "source_val"    - DataLoader over crossval splits of the 3 source domains
            "target_test"   - DataLoader over test split of the held-out domain
    """
    if target_domain not in PACS_DOMAINS:
        raise ValueError(
            f"target_domain '{target_domain}' not in PACS_DOMAINS: {PACS_DOMAINS}"
        )

    pacs_root = Path(config.root)
    label_dir = pacs_root / "pacs_label"
    img_root = pacs_root / "pacs_data" / "pacs_data"

    # Resolve source domains
    if config.domains:
        source_domains = [d for d in config.domains if d != target_domain]
    else:
        source_domains = [d for d in PACS_DOMAINS if d != target_domain]

    # Build default transforms
    if transform is None:
        transform = get_preprocessing_transform(model_name, is_training=False)
    if train_transform is None:
        train_transform = get_preprocessing_transform(model_name, is_training=True)

    # Collect samples per split
    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    for domain in source_domains:
        train_samples.extend(_parse_split_file(label_dir, img_root, domain, "train"))
        val_samples.extend(_parse_split_file(label_dir, img_root, domain, "crossval"))

    test_samples = _parse_split_file(label_dir, img_root, target_domain, "test")

    # Build datasets
    train_dataset = PACSDataset(train_samples, transform=train_transform)
    val_dataset = PACSDataset(val_samples, transform=transform)
    test_dataset = PACSDataset(test_samples, transform=transform)

    logger.info(
        f"PACS LODO | target={target_domain} | "
        f"source_train={len(train_dataset)} source_val={len(val_dataset)} "
        f"target_test={len(test_dataset)}"
    )

    worker_count = config.num_workers
    batch_size = config.batch_size
    pin = config.pin_memory

    source_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=pin,
        drop_last=True,
    )
    source_val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=pin,
        drop_last=False,
    )
    target_test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=pin,
        drop_last=False,
    )

    return {
        "source_train": source_train_loader,
        "source_val": source_val_loader,
        "target_test": target_test_loader,
    }
