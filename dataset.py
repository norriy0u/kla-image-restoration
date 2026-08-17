"""
Dataset Module for KLA Semiconductor Image Restoration
=====================================================
Handles paired .npy loading, speckle overflow normalization,
data augmentations, and test inference datasets.
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_noisy_lr(lr_arr: np.ndarray) -> np.ndarray:
    """
    Normalizes NoisyLR images containing speckle noise.
    Speckle noise pushes pixel values beyond [0, 1] up to ~2.0+.
    Clips to [-0.05, 2.0] and linearly scales to [0.0, 1.0].
    """
    clipped = np.clip(lr_arr, -0.05, 2.0)
    scaled = (clipped + 0.05) / 2.05
    return scaled.astype(np.float32)


def normalize_gt(gt_arr: np.ndarray) -> np.ndarray:
    """
    Normalizes Ground Truth clean images to [0.0, 1.0].
    """
    clipped = np.clip(gt_arr, 0.0, 1.0)
    return clipped.astype(np.float32)


class RestorationDataset(Dataset):
    """
    Paired dataset for training and validating NAFNet-SR model.
    GT shape: (256, 256)
    NoisyLR shape: (128, 128)
    """

    def __init__(
        self,
        gt_dir: str,
        lr_dir: str,
        split: str = "train",
        val_fraction: float = 0.1,
        patch_size_gt: int = 256,
        augment: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.split = split
        self.patch_size_gt = patch_size_gt
        self.augment = augment and (split == "train")

        # Find matching .npy files
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))

        if len(gt_files) == 0:
            raise FileNotFoundError(f"No .npy files found in GT directory: {gt_dir}")
        if len(lr_files) == 0:
            raise FileNotFoundError(f"No .npy files found in LR directory: {lr_dir}")

        # Pair up by file stem
        gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
        lr_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in lr_files}

        common_stems = sorted(list(set(gt_dict.keys()) & set(lr_dict.keys())))
        if len(common_stems) == 0:
            raise ValueError(f"No matching file stems between GT ({len(gt_files)}) and LR ({len(lr_files)})")

        # Split train / val deterministically
        rng = random.Random(seed)
        rng.shuffle(common_stems)

        val_size = int(len(common_stems) * val_fraction)
        if split == "val":
            self.stems = common_stems[:val_size]
        else:
            self.stems = common_stems[val_size:]

        self.gt_paths = [gt_dict[s] for s in self.stems]
        self.lr_paths = [lr_dict[s] for s in self.stems]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        gt_np = np.load(self.gt_paths[idx])
        lr_np = np.load(self.lr_paths[idx])

        gt_norm = normalize_gt(gt_np)
        lr_norm = normalize_noisy_lr(lr_np)

        # Augmentations for training split
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                gt_norm = np.fliplr(gt_norm).copy()
                lr_norm = np.fliplr(lr_norm).copy()
            # Random Vertical Flip
            if random.random() > 0.5:
                gt_norm = np.flipud(gt_norm).copy()
                lr_norm = np.flipud(lr_norm).copy()
            # Random 90-degree Rotation
            k = random.randint(0, 3)
            if k > 0:
                gt_norm = np.rot90(gt_norm, k).copy()
                lr_norm = np.rot90(lr_norm, k).copy()

        # Add channel dimension (C, H, W)
        gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0)  # (1, 256, 256)
        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0)  # (1, 128, 128)

        return lr_tensor, gt_tensor


class InferenceDataset(Dataset):
    """
    Dataset for test inference where Ground Truth is not available.
    """

    def __init__(self, input_dir: str):
        super().__init__()
        self.input_dir = input_dir
        self.file_paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .npy files found in input directory: {input_dir}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.file_paths[idx]
        filename = os.path.basename(path)
        lr_np = np.load(path)
        lr_norm = normalize_noisy_lr(lr_np)
        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0)
        return lr_tensor, filename
