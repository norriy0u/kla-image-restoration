"""
Composite Loss Function for Image Restoration
==============================================
L_total = λ_char * L_charbonnier
        + λ_freq * L_frequency
        + λ_ssim * L_ssim
        + λ_edge * L_edge

Design rationale:
  - Charbonnier: robust pixel-level reconstruction (handles speckle outliers better than L2)
  - Frequency:   FFT amplitude MSE — directly penalises loss of high-freq edge detail
  - SSIM:        Structural/perceptual quality
  - Edge (Sobel): Preserves sharp semiconductor feature boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: sqrt( (pred - target)^2 + eps^2 )
    Behaves like L1 for large errors, L2 for small errors.
    More robust than MSE to the outlier pixel values caused by speckle noise.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class FrequencyLoss(nn.Module):
    """
    Fourier-domain amplitude loss.

    Computes FFT of both pred and target, then measures MSE between their
    amplitude (magnitude) spectra. This directly penalises loss of
    high-frequency detail (edges, fine structures) — critical for
    semiconductor defect visibility.

    log(1 + |F|) is used to balance low vs high frequencies.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cast to float32: rfft2 does not support float16 on all CUDA versions
        pred_f   = pred.float()
        target_f = target.float()
        pred_fft   = torch.fft.rfft2(pred_f,   norm="ortho")
        target_fft = torch.fft.rfft2(target_f, norm="ortho")

        pred_amp   = torch.log1p(torch.abs(pred_fft))
        target_amp = torch.log1p(torch.abs(target_fft))

        return F.mse_loss(pred_amp, target_amp) * self.loss_weight


class SSIMLoss(nn.Module):
    """
    SSIM-based loss: 1 - SSIM(pred, target).
    Uses a Gaussian window. Window size 11 is standard.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._create_window(window_size, sigma))

    @staticmethod
    def _create_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        return window.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure same device and dtype as input (handles AMP float16 and CPU/GPU mismatch)
        x, y = x.float(), y.float()
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        w = self.window.to(device=x.device, dtype=x.dtype)

        mu_x = F.conv2d(x, w, padding=pad, groups=1)
        mu_y = F.conv2d(y, w, padding=pad, groups=1)

        mu_x2, mu_y2 = mu_x ** 2, mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_x  = F.conv2d(x * x, w, padding=pad, groups=1) - mu_x2
        sig_y  = F.conv2d(y * y, w, padding=pad, groups=1) - mu_y2
        sig_xy = F.conv2d(x * y, w, padding=pad, groups=1) - mu_xy

        ssim_map = (
            (2 * mu_xy + C1) * (2 * sig_xy + C2)
        ) / (
            (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
        )
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


class EdgeLoss(nn.Module):
    """
    Sobel edge loss: encourages the model to reproduce sharp edges.
    Computes Sobel gradient magnitude of both images, then L1 distance.
    Critical for preserving semiconductor feature boundaries.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        # Move kernels to same device and dtype as input
        x = x.float()
        kx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        ky = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._gradient_magnitude(pred), self._gradient_magnitude(target))


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    Weighted combination of all loss terms.

    Default weights (tuned for semiconductor restoration):
        λ_char = 1.0   — primary pixel reconstruction
        λ_freq = 0.1   — frequency-domain edges
        λ_ssim = 0.2   — perceptual / structural
        λ_edge = 0.1   — spatial gradient (Sobel)

    All weights are configurable via constructor or config dict.
    """

    def __init__(
        self,
        lambda_char: float = 1.0,
        lambda_freq: float = 0.1,
        lambda_ssim: float = 0.2,
        lambda_edge: float = 0.1,
    ):
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_freq = lambda_freq
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge

        self.char_loss = CharbonnierLoss()
        self.freq_loss = FrequencyLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor for .backward()
            breakdown:  dict of individual loss values for logging
        """
        l_char = self.char_loss(pred, target)
        l_freq = self.freq_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_edge = self.edge_loss(pred, target)

        total = (
            self.lambda_char * l_char
            + self.lambda_freq * l_freq
            + self.lambda_ssim * l_ssim
            + self.lambda_edge * l_edge
        )

        breakdown = {
            "char": l_char.item(),
            "freq": l_freq.item(),
            "ssim": l_ssim.item(),
            "edge": l_edge.item(),
            "total": total.item(),
        }
        return total, breakdown


if __name__ == "__main__":
    # Smoke test
    loss_fn = CompositeLoss()
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    total, breakdown = loss_fn(pred, target)
    print(f"Total loss: {total.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v:.4f}")
    print("✓ Loss function OK")
