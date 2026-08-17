"""
Training Script — NAFNet-SR for KLA Semiconductor Image Restoration
===================================================================
Usage:
    python train.py --gt_dir /path/to/GT --lr_dir /path/to/NoisyLR

    # Full example with defaults:
    python train.py \
        --gt_dir ~/Downloads/kla_extracted/train/train/GT \
        --lr_dir ~/Downloads/kla_extracted/train/train/NoisyLR \
        --epochs 200 \
        --batch_size 8 \
        --model_variant base \
        --weights_dir ./weights

    # Resume from checkpoint:
    python train.py ... --resume ./weights/checkpoint_epoch50.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure script root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from dataset import RestorationDataset
from model.nafnet import build_model
from model.losses import CompositeLoss
from utils.metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NAFNet-SR for KLA Image Restoration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument(
        "--gt_dir", type=str,
        default=os.path.expanduser("~/Downloads/kla_extracted/train/train/GT"),
        help="Path to GT directory (*.npy files, shape 256×256)"
    )
    parser.add_argument(
        "--lr_dir", type=str,
        default=os.path.expanduser("~/Downloads/kla_extracted/train/train/NoisyLR"),
        help="Path to NoisyLR directory (*.npy files, shape 128×128)"
    )
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of data held out for validation")
    parser.add_argument("--patch_size_gt", type=int, default=256,
                        help="GT crop size for training (LR crop = patch_size_gt//2)")

    # Model
    parser.add_argument("--model_variant", type=str, default="base",
                        choices=["tiny", "base", "large"])

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--lambda_char", type=float, default=1.0)
    parser.add_argument("--lambda_freq", type=float, default=0.1)
    parser.add_argument("--lambda_ssim", type=float, default=0.2)
    parser.add_argument("--lambda_edge", type=float, default=0.1)

    # Output
    parser.add_argument("--weights_dir", type=str, default="./weights")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt to resume from")
    parser.add_argument("--log_dir", type=str, default="./logs")

    # Hardware
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed-precision training")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# LR Scheduler (cosine with linear warmup)
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, use_amp, grad_clip):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            lr, gt = batch[0], batch[1]
        else:
            lr, gt = batch["lr"], batch["gt"]
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=use_amp):
            pred = model(lr)
            # Safety: resize if shape mismatch (shouldn't happen with correct data)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )
            loss, _ = loss_fn(pred, gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n += 1

    return total_loss / max(1, n)


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp):
    model.eval()
    tracker = MetricsTracker(track_lpips=False)
    total_loss = 0.0
    n = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            lr, gt = batch[0], batch[1]
        else:
            lr, gt = batch["lr"], batch["gt"]
        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=use_amp):
            pred = model(lr)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )
            loss, _ = loss_fn(pred, gt)

        tracker.update(pred.float(), gt.float(), batch_size=lr.shape[0])
        total_loss += loss.item()
        n += 1

    metrics = tracker.summary()
    metrics["loss"] = total_loss / max(1, n)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    print(f"[Train] Device: {device}  |  AMP: {use_amp}")
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Datasets
    train_ds = RestorationDataset(
        gt_dir=args.gt_dir, lr_dir=args.lr_dir,
        split="train", val_fraction=args.val_fraction,
        patch_size_gt=args.patch_size_gt, augment=True,
        seed=args.seed,
    )
    val_ds = RestorationDataset(
        gt_dir=args.gt_dir, lr_dir=args.lr_dir,
        split="val", val_fraction=args.val_fraction,
        patch_size_gt=None, augment=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = build_model(args.model_variant).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] NAFNet-SR ({args.model_variant}): {n_params/1e6:.2f}M parameters")

    # Loss
    loss_fn = CompositeLoss(
        lambda_char=args.lambda_char,
        lambda_freq=args.lambda_freq,
        lambda_ssim=args.lambda_ssim,
        lambda_edge=args.lambda_edge,
    )

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.9), weight_decay=1e-4,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr, args.lr
    )
    scaler = GradScaler(device='cuda', enabled=use_amp)

    # TensorBoard (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
        use_tb = True
        print(f"[Train] TensorBoard: tensorboard --logdir {args.log_dir}")
    except ImportError:
        writer, use_tb = None, False

    # Resume / Auto-resume
    start_epoch = 0
    best_ssim = 0.0
    
    if not args.resume and os.path.exists(args.weights_dir):
        import glob
        ckpts = sorted(
            glob.glob(os.path.join(args.weights_dir, "checkpoint_epoch*.pt")),
            key=lambda x: int(os.path.basename(x).replace("checkpoint_epoch", "").replace(".pt", "")) if os.path.basename(x).replace("checkpoint_epoch", "").replace(".pt", "").isdigit() else 0
        )
        if ckpts:
            args.resume = ckpts[-1]
            print(f"[Train] 🔄 Auto-resuming from latest checkpoint: {args.resume}")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ssim = ckpt.get("best_ssim", 0.0)
        print(f"[Train] Resumed from epoch {start_epoch} (best SSIM: {best_ssim:.4f})")

    # Training loop
    print(f"\n{'='*65}")
    print(f"  NAFNet-SR | {args.epochs} epochs | {len(train_ds)} train | {len(val_ds)} val")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, use_amp, args.grad_clip
        )
        scheduler.step()

        log_val = (epoch + 1) % 5 == 0 or epoch == args.epochs - 1

        if log_val:
            val_metrics = validate(model, val_loader, loss_fn, device, use_amp)
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:>3}/{args.epochs}] "
                f"lr={current_lr:.2e} | "
                f"train={train_loss:.4f} | "
                f"PSNR={val_metrics['psnr']:.2f}dB | "
                f"SSIM={val_metrics['ssim']:.4f} | "
                f"{elapsed:.1f}s"
            )
            if use_tb and writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Metrics/PSNR", val_metrics["psnr"], epoch)
                writer.add_scalar("Metrics/SSIM", val_metrics["ssim"], epoch)
                writer.add_scalar("LR", current_lr, epoch)

            # Save best
            if val_metrics["ssim"] > best_ssim:
                best_ssim = val_metrics["ssim"]
                best_path = os.path.join(args.weights_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_ssim": best_ssim,
                        "val_psnr": val_metrics["psnr"],
                        "args": vars(args),
                    },
                    best_path,
                )
                print(f"  ✓ Best model saved (SSIM={best_ssim:.4f})")
            import sys
            sys.stdout.flush()
        else:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:>3}/{args.epochs}] "
                f"lr={current_lr:.2e} | train={train_loss:.4f} | {elapsed:.1f}s"
            )
            import sys
            sys.stdout.flush()

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.weights_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_ssim": best_ssim,
                },
                ckpt_path,
            )
            print(f"  ✓ Checkpoint saved → {ckpt_path}")
            import sys
            sys.stdout.flush()

    print(f"\n[Train] Complete. Best val SSIM: {best_ssim:.4f}")
    if use_tb and writer:
        writer.close()


if __name__ == "__main__":
    main()
