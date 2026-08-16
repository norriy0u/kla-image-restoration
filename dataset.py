"""
Paired Dataset for KLA Semiconductor Image Restoration
=======================================================
Data layout (confirmed from actual dataset):

    train/train/
        GT/           ← 3200 files, .npy, shape (256, 256), float32, range [0, 1]
        NoisyLR/      ← 3200 files, .npy, shape (128, 128), float32, range [-0.003, ~1.5]

    test_noisyLR/
        NoisyLR/      ← 800 files, .npy, shape (128, 128), float32, range [0.001, ~1.54]

Key normalisation facts (confirmed from data inspection):
  - GT is always in [0, 1] — no clipping needed
  - NoisyLR CAN GO SLIGHTLY NEGATIVE (speckle) AND exceed 1.0
    → We clip to [-0.05, 2.0] then rescale to [0, 1]:
      x_norm = (clip(x, LO, HI) - LO) / (HI - LO)
  - Arrays are 2D (H, W) — we add a channel dim to get (1, H, W)
  - 198/200 NoisyLR files exceed 1.0 — speckle overflow is near-universal
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Clipping bounds for NoisyLR normalisation
# Data analysis: min ≈ -0.003, max ≈ 1.54 (test goes up to ~1.54)
# We use a small margin to be robust to unseen OOD extremes
NOISY_LO: float = -0.05
NOISY_HI: float = 2.00


def load_npy(path: str) -> torch.Tensor:
    """
    Load a .npy array and return as float32 tensor (1, H, W).
    Handles 2D (H, W) and 3D (H, W, C) or (C, H, W) arrays.
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]          # (H, W) → (1, H, W)
    elif arr.ndim == 3 and arr.shape[2] in (1, 3):
        arr = arr.transpose(2, 0, 1)        # (H, W, C) → (C, H, W)
    return torch.from_numpy(arr)


def normalise_noisy_lr(x: torch.Tensor) -> torch.Tensor:
    """
    Normalise NoisyLR input to [0, 1] for model consumption.

    Steps:
      1. Clip to [NOISY_LO, NOISY_HI]  — handles speckle overflow and negatives
      2. Rescale: (x - LO) / (HI - LO) — linear remap to [0, 1]
    """
    x = x.clamp(NOISY_LO, NOISY_HI)
    x = (x - NOISY_LO) / (NOISY_HI - NOISY_LO)
    return x


# ---------------------------------------------------------------------------
# Augmentation (applied identically to GT and NoisyLR pair)
# ---------------------------------------------------------------------------

def paired_augment(
    gt: torch.Tensor,
    lr: torch.Tensor,
    patch_size_gt: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply identical spatial augmentations to a (GT, NoisyLR) pair.

    All flips/rotations are applied using the same coin-flip to both tensors.
    Cropping uses the same top-left corner, scaled by the 2× factor between GT and LR.

    Args:
        gt:            (1, H,   W)   ground truth tensor [0, 1]
        lr:            (1, H/2, W/2) noisy LR tensor (already normalised)
        patch_size_gt: Crop GT to this size. LR crop = patch_size_gt // 2.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        gt = TF.hflip(gt)
        lr = TF.hflip(lr)

    # Random vertical flip
    if random.random() > 0.5:
        gt = TF.vflip(gt)
        lr = TF.vflip(lr)

    # Random 90° rotation (0, 90, 180, 270)
    if random.random() > 0.5:
        k = random.randint(1, 3)
        gt = torch.rot90(gt, k, dims=[-2, -1])
        lr = torch.rot90(lr, k, dims=[-2, -1])

    # Paired random crop
    if patch_size_gt is not None:
        _, H, W = gt.shape
        if H > patch_size_gt and W > patch_size_gt:
            top_gt = random.randint(0, H - patch_size_gt)
            left_gt = random.randint(0, W - patch_size_gt)
            # LR coordinates are exactly half (2× downsampling)
            top_lr = top_gt // 2
            left_lr = left_gt // 2
            patch_lr = patch_size_gt // 2

            gt = gt[:, top_gt:top_gt + patch_size_gt, left_gt:left_gt + patch_size_gt]
            lr = lr[:, top_lr:top_lr + patch_lr, left_lr:left_lr + patch_lr]

    return gt, lr


# ---------------------------------------------------------------------------
# Main training / validation dataset
# ---------------------------------------------------------------------------

class RestorationDataset(Dataset):
    """
    Paired GT / NoisyLR dataset for training and validation.

    Args:
        gt_dir:         Path to GT directory (contains *.npy files).
        lr_dir:         Path to NoisyLR directory (contains *.npy files).
        patch_size_gt:  GT patch size for random crop. None = full image.
        augment:        Apply random spatial augmentations (train only).
        val_fraction:   If > 0, split off this fraction as validation set.
        split:          'train' or 'val' (used with val_fraction).
        seed:           Random seed for train/val split.
        max_samples:    Cap dataset size (for quick experiments).
    """

    def __init__(
        self,
        gt_dir: str,
        lr_dir: str,
        patch_size_gt: Optional[int] = 256,
        augment: bool = True,
        val_fraction: float = 0.1,
        split: str = "train",
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.patch_size_gt = patch_size_gt
        self.augment = augment

        gt_dir = Path(gt_dir)
        lr_dir = Path(lr_dir)

        # Collect all matched pairs by stem name
        gt_files = sorted(gt_dir.glob("*.npy"))
        lr_stems = {p.stem: p for p in lr_dir.glob("*.npy")}

        all_pairs: List[Tuple[str, str]] = []
        for gt_path in gt_files:
            if gt_path.stem in lr_stems:
                all_pairs.append((str(gt_path), str(lr_stems[gt_path.stem])))

        if not all_pairs:
            raise RuntimeError(
                f"No matched pairs found between:\n  GT:  {gt_dir}\n  LR:  {lr_dir}\n"
                "Ensure both directories contain .npy files with matching names."
            )

        # Train/val split (deterministic)
        rng = random.Random(seed)
        shuffled = all_pairs.copy()
        rng.shuffle(shuffled)

        n_val = int(len(shuffled) * val_fraction)
        if split == "val":
            pairs = shuffled[:n_val]
            self.augment = False   # never augment validation
        else:
            pairs = shuffled[n_val:]

        if max_samples is not None:
            pairs = pairs[:max_samples]

        self.pairs = pairs
        print(
            f"[Dataset] {split}: {len(self.pairs)} pairs | "
            f"patch_size_gt={patch_size_gt} | augment={self.augment}"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        gt_path, lr_path = self.pairs[idx]

        gt = load_npy(gt_path)            # (1, 256, 256) in [0, 1]
        lr = load_npy(lr_path)            # (1, 128, 128) raw (may exceed [0,1])
        lr_norm = normalise_noisy_lr(lr)  # (1, 128, 128) normalised to [0, 1]

        if self.augment:
            gt, lr_norm = paired_augment(
                gt, lr_norm, patch_size_gt=self.patch_size_gt
            )

        return {
            "gt": gt,            # (1, H,   W)    target
            "lr": lr_norm,       # (1, H/2, W/2)  model input
            "gt_path": gt_path,
            "lr_path": lr_path,
        }


# ---------------------------------------------------------------------------
# Inference-only dataset (test set — no GT)
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """
    Single-directory dataset for test-set inference (no GT).

    Args:
        input_dir: Directory containing NoisyLR .npy files.
    """

    def __init__(self, input_dir: str):
        input_dir = Path(input_dir)
        self.paths = sorted(input_dir.glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found in: {input_dir}")
        print(f"[InferenceDataset] {len(self.paths)} images in {input_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        lr = load_npy(str(path))
        lr_norm = normalise_noisy_lr(lr)
        return {
            "lr": lr_norm,
            "path": str(path),
            "stem": path.stem,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    BASE = os.path.expanduser("~/Downloads/kla_extracted")
    GT_DIR = os.path.join(BASE, "train/train/GT")
    LR_DIR = os.path.join(BASE, "train/train/NoisyLR")
    TEST_DIR = os.path.join(BASE, "test_noisyLR/NoisyLR")

    # Training split
    ds_train = RestorationDataset(GT_DIR, LR_DIR, split="train", augment=True)
    ds_val   = RestorationDataset(GT_DIR, LR_DIR, split="val",   augment=False)
    print(f"\nTrain: {len(ds_train)} | Val: {len(ds_val)}")

    sample = ds_train[0]
    print(f"GT  shape: {sample['gt'].shape} | range [{sample['gt'].min():.3f}, {sample['gt'].max():.3f}]")
    print(f"LR  shape: {sample['lr'].shape} | range [{sample['lr'].min():.3f}, {sample['lr'].max():.3f}]")

    # Inference dataset
    ds_test = InferenceDataset(TEST_DIR)
    test_sample = ds_test[0]
    print(f"\nTest LR shape: {test_sample['lr'].shape} | range [{test_sample['lr'].min():.3f}, {test_sample['lr'].max():.3f}]")
    print("✓ All datasets OK")
