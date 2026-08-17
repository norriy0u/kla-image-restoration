"""
4-Panel Visual Demonstration & Interpretability Generator
========================================================
Generates a 4-panel visual comparison graphic:
  1. NoisyLR Input (128x128)
  2. ReDI-NAFNet Output (256x256)
  3. Ground Truth (256x256)
  4. Error Residual Heatmap (|Pred - GT|)
"""

import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 4-Panel Restoration Visual Comparison Plot")
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=os.path.expanduser("~/Downloads/train/train/GT"),
        help="Path to Ground Truth directory",
    )
    parser.add_argument(
        "--lr_dir",
        type=str,
        default=os.path.expanduser("~/Downloads/train/train/NoisyLR"),
        help="Path to NoisyLR input directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/val_outputs",
        help="Path to directory containing restored output .npy files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./outputs/before_after_comparison.png",
        help="Path to save comparison plot PNG",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Fallbacks for Windows local user downloads folder if default path not found
    gt_dir = args.gt_dir
    lr_dir = args.lr_dir
    if not os.path.exists(gt_dir):
        win_gt = r"C:\Users\wwwam\Downloads\train\train\GT"
        if os.path.exists(win_gt):
            gt_dir = win_gt
    if not os.path.exists(lr_dir):
        win_lr = r"C:\Users\wwwam\Downloads\train\train\NoisyLR"
        if os.path.exists(win_lr):
            lr_dir = win_lr

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))

    if len(lr_files) == 0:
        print(f"[Error] No .npy files found in LR directory: {lr_dir}")
        return

    # Pick first 3 stems where output file exists
    sample_stems = []
    for f in lr_files[:200]:
        stem = os.path.splitext(os.path.basename(f))[0]
        out_npy = os.path.join(args.output_dir, f"{stem}.npy")
        gt_npy = os.path.join(gt_dir, f"{stem}.npy")
        if os.path.exists(out_npy) and os.path.exists(gt_npy):
            sample_stems.append(stem)
            if len(sample_stems) >= 3:
                break

    if len(sample_stems) == 0:
        print(f"[Error] No matching restored output files found in: {args.output_dir}")
        return

    fig, axes = plt.subplots(len(sample_stems), 4, figsize=(18, 13))
    if len(sample_stems) == 1:
        axes = [axes]

    for row, stem in enumerate(sample_stems):
        lr_arr = np.load(os.path.join(lr_dir, f"{stem}.npy"))
        gt_arr = np.load(os.path.join(gt_dir, f"{stem}.npy"))
        pr_arr = np.load(os.path.join(args.output_dir, f"{stem}.npy"))

        # Residual error map (|Pred - GT|)
        residual = np.abs(pr_arr - gt_arr)

        # 1. NoisyLR
        axes[row][0].imshow(np.clip(lr_arr, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(
            f"NoisyLR #{stem}\n[{lr_arr.min():.2f}, {lr_arr.max():.2f}]",
            fontsize=10,
            color="#e74c3c",
            fontweight="bold",
        )
        axes[row][0].axis("off")

        # 2. Restored
        axes[row][1].imshow(np.clip(pr_arr, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[row][1].set_title(
            f"ReDI-NAFNet Output\n[{pr_arr.min():.2f}, {pr_arr.max():.2f}]",
            fontsize=10,
            color="#2ecc71",
            fontweight="bold",
        )
        axes[row][1].axis("off")

        # 3. Ground Truth
        axes[row][2].imshow(np.clip(gt_arr, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[row][2].set_title(
            f"Ground Truth\n[{gt_arr.min():.2f}, {gt_arr.max():.2f}]",
            fontsize=10,
            color="#3498db",
            fontweight="bold",
        )
        axes[row][2].axis("off")

        # 4. Error Residual Heatmap (|Pred - GT|)
        im = axes[row][3].imshow(residual, cmap="inferno", vmin=0, vmax=0.3)
        axes[row][3].set_title(
            f"Error Residual Map (|Pred-GT|)\nMean Error: {residual.mean():.4f}",
            fontsize=10,
            color="#9b59b6",
            fontweight="bold",
        )
        axes[row][3].axis("off")

    plt.suptitle("ReDI-NAFNet: 4-Panel Semiconductor Image Restoration & Residual Error Map", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")

    # Secondary save to Downloads if available
    downloads_save = os.path.expanduser("~/Downloads/before_after_comparison.png")
    try:
        plt.savefig(downloads_save, dpi=150, bbox_inches="tight")
    except Exception:
        pass

    plt.close()
    print(f"[Success] Saved 4-panel comparison graphic to: {args.save_path}")


if __name__ == "__main__":
    main()
