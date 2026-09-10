"""
Mini-Ecoset 112-Class HDF5 Dataset and DataModule for DynVision.

Supports:
- texture2shape_miniecoset_256px.h5 (302,400 images, 112 categories)
- Standard splits: 'train' (263,200), 'val' (5,600), 'test' (5,600), 'testplus' (28,000)
- Zero-overhead SWMR HDF5 reading and optional in-memory caching
- Dataset-specific channel normalization from container metadata
- Geirhos-16 Cue-Conflict category mapping for model-vs-human benchmark
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Disable HDF5 file locking to allow seamless cross-platform reading over network/WSL shares
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# Default empirical channel statistics computed from texture2shape_miniecoset_256px.h5
MINIECOSET_MEAN = [0.49276, 0.46849, 0.40767]
MINIECOSET_STD = [0.28310, 0.27594, 0.29276]

# The 16 Geirhos Model-vs-Human Cue-Conflict benchmark classes
GEIRHOS_16_CLASSES = [
    "airplane", "bear", "bicycle", "bird", "boat", "bottle",
    "car", "cat", "chair", "clock", "dog", "elephant",
    "keyboard", "knife", "oven", "truck"
]

# Alias dictionary between Ecoset class naming and Geirhos class naming
ECOSET_TO_GEIRHOS_ALIASES = {
    "typewriter": "keyboard",
    "keyboard": "keyboard",
    "stove": "oven",
    "oven": "oven",
}

GEIRHOS_TO_ECOSET_ALIASES = {
    "keyboard": "typewriter",
    "oven": "stove",
}


def get_default_transforms(
    img_size: int = 224,
    split: str = "train",
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Callable:
    """Build standard PyTorch transforms for Mini-Ecoset training and validation."""
    mean = mean or MINIECOSET_MEAN
    std = std or MINIECOSET_STD

    if split == "train":
        return T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


class MiniEcosetH5Dataset(Dataset):
    """
    High-performance PyTorch Dataset for 112-class Mini-Ecoset HDF5 container.
    """

    def __init__(
        self,
        h5_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        img_size: int = 224,
        in_memory: bool = False,
    ):
        super().__init__()
        if h5_path is None:
            if sys.platform == "win32":
                h5_path = Path(r"\\wsl.localhost\Ubuntu\home\glenn\DynVision_Working\data\raw\mini-ecoset\texture2shape_miniecoset_256px.h5")
            else:
                h5_path = Path("/home/glenn/DynVision_Working/data/raw/mini-ecoset/texture2shape_miniecoset_256px.h5")

        self.h5_path = Path(h5_path)
        if self.h5_path.is_dir():
            # Auto-detect H5 file in directory
            candidate = self.h5_path / "texture2shape_miniecoset_256px.h5"
            if candidate.exists():
                self.h5_path = candidate
            else:
                h5_files = list(self.h5_path.glob("*.h5"))
                if not h5_files:
                    raise FileNotFoundError(f"No .h5 file found in {self.h5_path}")
                self.h5_path = h5_files[0]

        if not self.h5_path.exists():
            raise FileNotFoundError(f"Mini-Ecoset H5 container not found at {self.h5_path}")

        self.split = split.lower()
        self.target_transform = target_transform
        self.in_memory = in_memory
        self._h5 = None

        # Read metadata and pre-validate split
        with h5py.File(self.h5_path, "r") as f:
            if self.split not in f:
                raise ValueError(f"Split '{self.split}' not in {list(f.keys())}")

            self._len = int(f[self.split]["labels"].shape[0])
            self.image_shape = f[self.split]["data"].shape[1:]

            # Read categories
            if "categories" in f:
                raw_cats = f["categories"][:]
                self.categories = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in raw_cats]
            else:
                self.categories = [f"class_{i:03d}" for i in range(112)]

            # Load channel statistics if available
            if "train_img_mean_channels" in f and "train_img_std_channels" in f:
                self.mean = list(f["train_img_mean_channels"][:] / 255.0)
                self.std = list(f["train_img_std_channels"][:] / 255.0)
            else:
                self.mean = MINIECOSET_MEAN
                self.std = MINIECOSET_STD

        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_default_transforms(
                img_size=img_size, split=self.split, mean=self.mean, std=self.std
            )

        # In-memory caching for evaluation splits
        if self.in_memory:
            logger.info(f"Loading split '{self.split}' ({self._len} images) into RAM...")
            with h5py.File(self.h5_path, "r") as f:
                self._cached_data = f[self.split]["data"][:]
                self._cached_labels = f[self.split]["labels"][:]
        else:
            self._cached_data = None
            self._cached_labels = None

    def _init_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            self._data_ds = self._h5[self.split]["data"]
            self._labels_ds = self._h5[self.split]["labels"]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.in_memory:
            img_arr = self._cached_data[idx]
            label = int(self._cached_labels[idx])
        else:
            self._init_h5()
            img_arr = self._data_ds[idx]
            label = int(self._labels_ds[idx])

        # img_arr shape: (256, 256, 3), dtype uint8
        if self.transform is not None:
            img = self.transform(img_arr)
        else:
            img = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def get_geirhos_mapping(self) -> Dict[str, Any]:
        """
        Returns mapping from the 16 Geirhos cue-conflict classes to 112 Mini-Ecoset class indices.
        """
        return get_geirhos_mapping(self.categories)


def get_geirhos_mapping(categories_112: List[str]) -> Dict[str, Any]:
    """
    Constructs an index projection between the 16 Geirhos categories and the 112 Mini-Ecoset classes.

    Returns:
        dict containing:
            - 'geirhos_to_112_indices': List[int] of length 16 with corresponding 112-class indices
            - 'geirhos_classes': List[str] of the 16 Geirhos class names
            - 'ecoset_matched_names': List[str] of matched class names in the 112 classes
    """
    indices = []
    matched_names = []
    for g_cat in GEIRHOS_16_CLASSES:
        target_name = GEIRHOS_TO_ECOSET_ALIASES.get(g_cat, g_cat)
        if target_name in categories_112:
            idx = categories_112.index(target_name)
        elif g_cat in categories_112:
            idx = categories_112.index(g_cat)
        else:
            raise ValueError(
                f"Category '{g_cat}' (alias '{target_name}') not found in 112 classes: {categories_112}"
            )
        indices.append(idx)
        matched_names.append(target_name)

    return {
        "geirhos_to_112_indices": indices,
        "geirhos_classes": GEIRHOS_16_CLASSES,
        "ecoset_matched_names": matched_names,
    }


class MiniEcosetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for 112-Class Mini-Ecoset.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        img_size: int = 224,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        in_memory_val: bool = False,
    ):
        super().__init__()
        if data_path is None:
            if sys.platform == "win32":
                data_path = Path(r"\\wsl.localhost\Ubuntu\home\glenn\DynVision_Working\data\raw\mini-ecoset\texture2shape_miniecoset_256px.h5")
            else:
                data_path = Path("/home/glenn/DynVision_Working/data/raw/mini-ecoset/texture2shape_miniecoset_256px.h5")
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.in_memory_val = in_memory_val

        self.train_dataset: Optional[MiniEcosetH5Dataset] = None
        self.val_dataset: Optional[MiniEcosetH5Dataset] = None
        self.test_dataset: Optional[MiniEcosetH5Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = MiniEcosetH5Dataset(
                h5_path=self.data_path,
                split="train",
                img_size=self.img_size,
                in_memory=False,
            )
            self.val_dataset = MiniEcosetH5Dataset(
                h5_path=self.data_path,
                split="val",
                img_size=self.img_size,
                in_memory=self.in_memory_val,
            )
            logger.info(
                f"MiniEcoset DataModule setup fit: {len(self.train_dataset)} train, {len(self.val_dataset)} val"
            )

        if stage in (None, "test"):
            self.test_dataset = MiniEcosetH5Dataset(
                h5_path=self.data_path,
                split="test",
                img_size=self.img_size,
                in_memory=self.in_memory_val,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
