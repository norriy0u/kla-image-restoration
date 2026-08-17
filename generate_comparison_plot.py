import os
import glob
import numpy as np
import matplotlib.pyplot as plt

GT_DIR = r"C:\Users\wwwam\Downloads\train\train\GT"
LR_DIR = r"C:\Users\wwwam\Downloads\train\train\NoisyLR"
OUT_DIR = r"C:\Users\wwwam\.gemini\antigravity-ide\scratch\kla_restoration\outputs\val_outputs"

gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.npy")))
lr_files = sorted(glob.glob(os.path.join(LR_DIR, "*.npy")))

# Pick 3 interesting sample stems
sample_stems = []
for f in lr_files[:100]:
    stem = os.path.splitext(os.path.basename(f))[0]
    out_npy = os.path.join(OUT_DIR, f"{stem}.npy")
    gt_npy = os.path.join(GT_DIR, f"{stem}.npy")
    if os.path.exists(out_npy) and os.path.exists(gt_npy):
        sample_stems.append(stem)
        if len(sample_stems) >= 3:
            break

fig, axes = plt.subplots(len(sample_stems), 4, figsize=(18, 13))
col_titles = [
    "1. Degraded Input (128x128)",
    "2. ReDI-NAFNet Output (256x256)",
    "3. Ground Truth (256x256)",
    "4. Confidence / Error Residual Map",
]

for row, stem in enumerate(sample_stems):
    lr_arr = np.load(os.path.join(LR_DIR, f"{stem}.npy"))
    gt_arr = np.load(os.path.join(GT_DIR, f"{stem}.npy"))
    pr_arr = np.load(os.path.join(OUT_DIR, f"{stem}.npy"))

    # Residual error map (|Pred - GT|)
    residual = np.abs(pr_arr - gt_arr)

    # 1. NoisyLR
    axes[row][0].imshow(np.clip(lr_arr, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[row][0].set_title(f"NoisyLR #{stem}\n[{lr_arr.min():.2f}, {lr_arr.max():.2f}]", fontsize=10, color="#e74c3c", fontweight="bold")
    axes[row][0].axis("off")

    # 2. Restored
    axes[row][1].imshow(np.clip(pr_arr, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[row][1].set_title(f"Restored Output\n[{pr_arr.min():.2f}, {pr_arr.max():.2f}]", fontsize=10, color="#2ecc71", fontweight="bold")
    axes[row][1].axis("off")

    # 3. Ground Truth
    axes[row][2].imshow(np.clip(gt_arr, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[row][2].set_title(f"Ground Truth\n[{gt_arr.min():.2f}, {gt_arr.max():.2f}]", fontsize=10, color="#3498db", fontweight="bold")
    axes[row][2].axis("off")

    # 4. Residual Heatmap
    im = axes[row][3].imshow(residual, cmap="inferno", vmin=0, vmax=0.3)
    axes[row][3].set_title(f"Error Residual (|Pred-GT|)\nMean Error: {residual.mean():.4f}", fontsize=10, color="#9b59b6", fontweight="bold")
    axes[row][3].axis("off")

plt.suptitle("ReDI-NAFNet: 4-Panel Semiconductor Image Restoration & Interpretability", fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout()

save_path_1 = r"C:\Users\wwwam\.gemini\antigravity-ide\scratch\kla_restoration\outputs\before_after_comparison.png"
save_path_2 = r"C:\Users\wwwam\Downloads\before_after_comparison.png"

plt.savefig(save_path_1, dpi=150, bbox_inches="tight")
plt.savefig(save_path_2, dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ Saved 4-panel comparison graphic to:\n  - {save_path_1}\n  - {save_path_2}")
