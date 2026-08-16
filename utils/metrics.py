"""
Image Quality Metrics for Restoration Evaluation
=================================================
Implements:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
  - MetricsTracker — running average accumulator for training loops

All metrics expect float32 tensors in [0, 1] range, shape (B, C, H, W).
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB). Higher is better.

    Args:
        pred:    (B, C, H, W) float tensor in [0, 1]
        target:  (B, C, H, W) float tensor in [0, 1]
        max_val: Maximum pixel value (1.0 for normalised images)

    Returns:
        Mean PSNR across the batch (float, dB).
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
    return psnr.mean().item()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    return window.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    Structural Similarity Index (SSIM). Higher is better, max 1.0.

    Args:
        pred:       (B, C, H, W) float tensor
        target:     (B, C, H, W) float tensor
        window_size: Gaussian window size (default 11)
        sigma:       Gaussian sigma (default 1.5)
        data_range:  Pixel value range (default 1.0)

    Returns:
        Mean SSIM across the batch (float).
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = window_size // 2

    window = _gaussian_window(window_size, sigma, device=pred.device)
    B, C, H, W = pred.shape

    # Process each channel separately
    ssim_vals = []
    for c in range(C):
        x = pred[:, c:c+1, :, :]
        y = target[:, c:c+1, :, :]

        mu_x = F.conv2d(x, window, padding=pad)
        mu_y = F.conv2d(y, window, padding=pad)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_x = F.conv2d(x * x, window, padding=pad) - mu_x2
        sig_y = F.conv2d(y * y, window, padding=pad) - mu_y2
        sig_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

        ssim_map = (
            (2 * mu_xy + C1) * (2 * sig_xy + C2)
        ) / (
            (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
        )
        ssim_vals.append(ssim_map.mean())

    return torch.stack(ssim_vals).mean().item()


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

_lpips_model = None


def _get_lpips():
    """Lazy-load LPIPS to avoid import overhead when not needed."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net="alex", verbose=False)
            _lpips_model.eval()
        except ImportError:
            raise ImportError(
                "lpips package not found. Install with: pip install lpips"
            )
    return _lpips_model


def compute_lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS). Lower is better.

    LPIPS requires 3-channel input — we repeat grayscale channel 3×.
    Returns mean LPIPS across the batch.

    Args:
        pred:   (B, 1, H, W) or (B, 3, H, W) float tensor in [0, 1]
        target: same shape as pred

    Returns:
        Mean LPIPS score (float). Lower = more perceptually similar.
    """
    lpips_fn = _get_lpips()

    # Ensure device matches
    device = pred.device
    lpips_fn = lpips_fn.to(device)

    # LPIPS expects [0,1] → rescale to [-1, 1]
    p = pred * 2.0 - 1.0
    t = target * 2.0 - 1.0

    # Repeat grayscale to 3 channels
    if p.shape[1] == 1:
        p = p.repeat(1, 3, 1, 1)
        t = t.repeat(1, 3, 1, 1)

    with torch.no_grad():
        score = lpips_fn(p, t)

    return score.mean().item()


# ---------------------------------------------------------------------------
# MetricsTracker — running average for training loops
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Accumulates PSNR, SSIM (and optionally LPIPS) over an epoch.

    Usage:
        tracker = MetricsTracker()
        for batch in dataloader:
            pred, target = model(batch['lr']), batch['gt']
            tracker.update(pred, target, batch_size=pred.shape[0])
        stats = tracker.summary()
        print(stats)
    """

    def __init__(self, track_lpips: bool = False):
        self.track_lpips = track_lpips
        self.reset()

    def reset(self):
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._lpips_sum = 0.0
        self._n = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch_size: Optional[int] = None,
    ):
        """Update running averages with a new batch."""
        n = batch_size or pred.shape[0]
        self._psnr_sum += compute_psnr(pred, target) * n
        self._ssim_sum += compute_ssim(pred, target) * n
        if self.track_lpips:
            self._lpips_sum += compute_lpips(pred, target) * n
        self._n += n

    def summary(self) -> dict:
        """Return mean metrics over all accumulated batches."""
        if self._n == 0:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        result = {
            "psnr": self._psnr_sum / self._n,
            "ssim": self._ssim_sum / self._n,
        }
        if self.track_lpips:
            result["lpips"] = self._lpips_sum / self._n
        return result

    def __repr__(self) -> str:
        s = self.summary()
        parts = [f"PSNR={s['psnr']:.2f}dB", f"SSIM={s['ssim']:.4f}"]
        if "lpips" in s:
            parts.append(f"LPIPS={s['lpips']:.4f}")
        return " | ".join(parts)


if __name__ == "__main__":
    pred = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)

    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

    tracker = MetricsTracker()
    tracker.update(pred, target)
    print(f"Tracker: {tracker}")
    print("✓ Metrics OK")
