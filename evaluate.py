"""
Standalone Evaluation Script — KLA Semiconductor Image Restoration
===================================================================
CRITICAL: This script is used AS-IS by KLA benchmarking team.
          It must run without any manual edits.

Usage:
    # Inference only (no GT — official test set):
    python evaluate.py --input_dir /path/to/NoisyLR --output_dir /path/to/outputs

    # Inference + metrics (if you have GT):
    python evaluate.py --input_dir /path/to/NoisyLR --output_dir /path/to/outputs \
                       --gt_dir /path/to/GT

    # Custom weights:
    python evaluate.py --input_dir /path/to/NoisyLR --output_dir /path/to/outputs \
                       --weights /path/to/best_model.pt

Arguments:
    --input_dir   (required) Path to directory containing NoisyLR .npy files
    --output_dir  (required) Path to save restored images (.npy and .png)
    --weights     Path to model weights .pt file (default: ./weights/best_model.pt)
    --gt_dir      Path to GT directory for metric computation (optional)
    --device      'cuda' or 'cpu' (default: auto-detect)
    --batch_size  Inference batch size (default: 4)
    --tta         Enable test-time augmentation (4 flips, default: off)
    --variant     Model variant: tiny / base / large (default: base)
    --save_png    Also save outputs as .png images (default: True)

Output:
    - <output_dir>/<stem>.npy  — restored array, shape (256,256), float32, range [0,1]
    - <output_dir>/<stem>.png  — 8-bit PNG for visual inspection
    - <output_dir>/metrics.json — PSNR/SSIM/LPIPS summary (if --gt_dir provided)
    - Console: per-image and aggregate metrics + total inference time
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path so this script works from any directory
sys.path.insert(0, str(Path(__file__).parent))

from dataset import InferenceDataset, load_npy, normalise_noisy_lr
from model.nafnet import build_model
from utils.metrics import compute_psnr, compute_ssim

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "best_model.pt")


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA)
# ---------------------------------------------------------------------------

def tta_predict(model: torch.nn.Module, lr: torch.Tensor) -> torch.Tensor:
    """
    4-fold TTA: identity + h-flip + v-flip + hv-flip.
    Averages predictions in the original (unflipped) space.
    """
    preds = []
    for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
        x = lr.clone()
        if flip_h:
            x = torch.flip(x, dims=[-1])
        if flip_v:
            x = torch.flip(x, dims=[-2])

        with torch.no_grad():
            p = model(x)

        if flip_h:
            p = torch.flip(p, dims=[-1])
        if flip_v:
            p = torch.flip(p, dims=[-2])

        preds.append(p)

    return torch.stack(preds, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="KLA Image Restoration — Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to directory containing NoisyLR .npy files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save restored images")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="(Optional) GT directory for metric computation")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Compute device")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (slightly slower, better quality)")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["tiny", "base", "large"],
                        help="Model variant (must match trained weights)")
    parser.add_argument("--no_png", action="store_true",
                        help="Skip saving .png outputs (saves only .npy)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Evaluate] Device: {device}")

    # ---- Directories ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ----
    print(f"[Evaluate] Loading model variant='{args.variant}' from: {args.weights}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(
            f"Model weights not found: {args.weights}\n"
            "Download from the repository and place at: weights/best_model.pt"
        )

    model = build_model(args.variant).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    # Support both raw state_dict and checkpoint dict formats
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Evaluate] Model params: {n_params/1e6:.2f}M")
    print(f"[Evaluate] TTA: {args.tta}")

    # ---- Dataset ----
    dataset = InferenceDataset(args.input_dir)
    # Use batch_size=1 for TTA (avoids complexity), else batch for speed
    effective_bs = 1 if args.tta else args.batch_size
    loader = DataLoader(
        dataset,
        batch_size=effective_bs,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    # ---- GT loader (optional, for metrics) ----
    gt_map = {}
    if args.gt_dir is not None:
        gt_dir = Path(args.gt_dir)
        for p in gt_dir.glob("*.npy"):
            gt_map[p.stem] = str(p)
        print(f"[Evaluate] GT dir: {gt_dir} ({len(gt_map)} files)")
        compute_metrics = len(gt_map) > 0
    else:
        compute_metrics = False

    # ---- Inference ----
    print(f"\n[Evaluate] Running inference on {len(dataset)} images...")
    print("-" * 60)

    all_psnr, all_ssim = [], []
    total_time = 0.0
    save_png = not args.no_png

    try:
        from PIL import Image as PILImage
        has_pil = True
    except ImportError:
        has_pil = False
        if save_png:
            print("[Warning] Pillow not installed — skipping .png output")
            save_png = False

    for batch_idx, batch in enumerate(loader):
        lr = batch["lr"].to(device, non_blocking=True)
        stems = batch["stem"]
        paths = batch["path"]

        t0 = time.perf_counter()
        with torch.no_grad():
            if args.tta:
                pred = tta_predict(model, lr)
            else:
                pred = model(lr)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        pred_np = pred.cpu().float().numpy()  # (B, 1, 256, 256)

        for i, stem in enumerate(stems):
            arr = pred_np[i, 0]  # (256, 256) float32 [0, 1]

            # Save .npy
            out_npy = os.path.join(args.output_dir, f"{stem}.npy")
            np.save(out_npy, arr)

            # Save .png
            if save_png and has_pil:
                png_arr = (arr * 255).clip(0, 255).astype(np.uint8)
                out_png = os.path.join(args.output_dir, f"{stem}.png")
                PILImage.fromarray(png_arr, mode="L").save(out_png)

            # Metrics
            if compute_metrics and stem in gt_map:
                gt_arr = np.load(gt_map[stem]).astype(np.float32)
                gt_t = torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                pr_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

                psnr = compute_psnr(pr_t, gt_t)
                ssim = compute_ssim(pr_t, gt_t)
                all_psnr.append(psnr)
                all_ssim.append(ssim)

                if (batch_idx * effective_bs + i) % 50 == 0:
                    print(f"  [{stem}] PSNR={psnr:.2f}dB | SSIM={ssim:.4f} | time={elapsed*1000:.1f}ms")
            elif batch_idx % 50 == 0:
                print(f"  [{stem}] restored | time={elapsed*1000:.1f}ms/batch")

    # ---- Summary ----
    n = len(dataset)
    avg_time_ms = total_time / max(1, n) * 1000
    print("\n" + "=" * 60)
    print(f"  Total images:     {n}")
    print(f"  Total time:       {total_time:.2f}s")
    print(f"  Avg per image:    {avg_time_ms:.1f}ms")
    print(f"  Throughput:       {n / max(total_time, 1e-6):.1f} images/sec")

    results = {
        "n_images": n,
        "total_time_s": round(total_time, 3),
        "avg_time_ms": round(avg_time_ms, 2),
        "throughput_imgs_per_sec": round(n / max(total_time, 1e-6), 2),
        "device": str(device),
        "tta": args.tta,
        "variant": args.variant,
    }

    if compute_metrics and all_psnr:
        mean_psnr = float(np.mean(all_psnr))
        mean_ssim = float(np.mean(all_ssim))
        print(f"\n  Mean PSNR:        {mean_psnr:.4f} dB")
        print(f"  Mean SSIM:        {mean_ssim:.4f}")
        results["psnr"] = round(mean_psnr, 4)
        results["ssim"] = round(mean_ssim, 4)

        # Optionally compute LPIPS
        try:
            from utils.metrics import compute_lpips
            # Load all preds/gts into memory for LPIPS (may be slow for large sets)
            print("\n  Computing LPIPS (this may take a moment)...")
            all_lpips = []
            for f in Path(args.input_dir).glob("*.npy"):
                stem = f.stem
                if stem not in gt_map:
                    continue
                out_npy = os.path.join(args.output_dir, f"{stem}.npy")
                if not os.path.exists(out_npy):
                    continue
                pr_t = torch.from_numpy(np.load(out_npy)).unsqueeze(0).unsqueeze(0)
                gt_t = torch.from_numpy(np.load(gt_map[stem]).astype(np.float32)).unsqueeze(0).unsqueeze(0)
                all_lpips.append(compute_lpips(pr_t, gt_t))
            if all_lpips:
                mean_lpips = float(np.mean(all_lpips))
                print(f"  Mean LPIPS:       {mean_lpips:.4f}")
                results["lpips"] = round(mean_lpips, 4)
        except Exception as e:
            print(f"  LPIPS skipped: {e}")

    print("=" * 60)

    # Save results JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Evaluate] Results saved to: {metrics_path}")
    print(f"[Evaluate] Outputs saved to:  {args.output_dir}")


if __name__ == "__main__":
    main()
