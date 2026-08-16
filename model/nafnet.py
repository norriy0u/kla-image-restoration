"""
NAFNet-SR: Nonlinear Activation Free Network with Super-Resolution Head
==========================================================================
Architecture: U-Net encoder-decoder with NAFBlocks + PixelShuffle SR head.

Reference: "Simple Baselines for Image Restoration" (ECCV 2022)
           https://arxiv.org/abs/2204.04676

Key design choices:
  - SimpleGate replaces all nonlinear activations
  - LayerNorm2d for better generalization than BatchNorm
  - Depthwise conv in NAFBlock for efficiency
  - PixelShuffle ×2 upscaling at the output head
  - Single forward pass handles denoising + super-resolution jointly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalisation for 2D feature maps (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies them — replaces all activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ---------------------------------------------------------------------------
# Core NAFBlock
# ---------------------------------------------------------------------------

class NAFBlock(nn.Module):
    """
    NAFNet building block.

    Structure:
        Branch 1 (spatial mixing):
            LN → Conv1×1 (expand) → DW Conv3×3 → SimpleGate → SCA → Conv1×1 (squeeze)
        Branch 2 (channel mixing / FFN):
            LN → Conv1×1 (expand) → SimpleGate → Conv1×1 (squeeze)
        Both branches use learnable residual scaling (β, γ).
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch = c * dw_expand

        # Branch 1 — spatial
        self.conv1 = nn.Conv2d(c, dw_ch, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=True)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1, bias=True),
        )

        # Branch 2 — FFN
        ffn_ch = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_ch, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gate = SimpleGate()

        # Learnable residual scaling — initialised small so training is stable
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-3)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Branch 1
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        # Branch 2 (FFN)
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gate(x)
        x = self.conv5(x)
        return y + x * self.gamma


# ---------------------------------------------------------------------------
# Encoder / Decoder utilities
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """Pixel-unshuffle based downsampling (lossless, fast)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(2),        # (B, C, H, W) → (B, 4C, H/2, W/2)
            nn.Conv2d(c_in * 4, c_out, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpBlock(nn.Module):
    """Pixel-shuffle based upsampling."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(c_in, c_out * 4, 1, bias=False),
            nn.PixelShuffle(2),          # (B, 4C, H, W) → (B, C, 2H, 2W)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ---------------------------------------------------------------------------
# NAFNet-SR main model
# ---------------------------------------------------------------------------

class NAFNetSR(nn.Module):
    """
    NAFNet with a 2× Super-Resolution output head.

    Input:  (B, 1, H,   W)   — NoisyLR, normalised and clipped [0, 1.5]
    Output: (B, 1, 2H, 2W)   — Restored full-resolution image, clamped [0, 1]

    Args:
        in_ch:      Input channels (1 for grayscale).
        width:      Base channel width (default 32).  Scales: 32/64/128/256.
        enc_blocks: NAFBlocks per encoder level.
        dec_blocks: NAFBlocks per decoder level.
        mid_blocks: NAFBlocks in the bottleneck.
    """

    def __init__(
        self,
        in_ch: int = 1,
        width: int = 32,
        enc_blocks: list[int] = [2, 2, 4, 8],
        dec_blocks: list[int] = [2, 2, 2, 2],
        mid_blocks: int = 4,
    ):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        enc_channels = []
        for n_blocks in enc_blocks:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch) for _ in range(n_blocks)])
            )
            enc_channels.append(ch)
            self.downs.append(DownBlock(ch, ch * 2))
            ch = ch * 2

        # ---- Bottleneck ----
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(mid_blocks)])

        # ---- Decoder ----
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for n_blocks, skip_ch in zip(dec_blocks, reversed(enc_channels)):
            self.ups.append(UpBlock(ch, ch // 2))
            ch = ch // 2
            # After concatenation with skip: ch + skip_ch = 2*ch (since skip_ch == ch)
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(ch + skip_ch, ch, 1, bias=True),
                    *[NAFBlock(ch) for _ in range(n_blocks)],
                )
            )

        # ---- SR Head (×2 upscale via PixelShuffle) ----
        self.sr_head = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1, bias=True),  # expand for shuffle
            nn.PixelShuffle(2),                                # → (B, ch, 2H, 2W)
            nn.Conv2d(ch, in_ch, 3, padding=1, bias=True),    # → (B, 1, 2H, 2W)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: NoisyLR tensor (B, 1, H, W), pre-normalised to [0, 1]
        Returns:
            Restored image (B, 1, 2H, 2W), clamped [0, 1]
        """
        inp = self.intro(x)

        # Encoder
        enc_feats = []
        feat = inp
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            enc_feats.append(feat)
            feat = down(feat)

        # Bottleneck
        feat = self.middle(feat)

        # Decoder
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(enc_feats)):
            feat = up(feat)
            feat = decoder(torch.cat([feat, skip], dim=1))

        # SR upscale + residual (bilinear upsampled input)
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.sr_head(feat) + x_up
        return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(variant: str = "base") -> NAFNetSR:
    """
    Predefined model variants.

    - 'tiny'  : fast inference, smaller capacity (good for testing)
    - 'base'  : recommended for submission
    - 'large' : maximum quality, slower
    """
    configs = {
        "tiny": dict(width=16, enc_blocks=[1, 1, 2, 2], dec_blocks=[1, 1, 1, 1], mid_blocks=2),
        "base": dict(width=32, enc_blocks=[2, 2, 4, 8], dec_blocks=[2, 2, 2, 2], mid_blocks=4),
        "large": dict(width=64, enc_blocks=[2, 2, 4, 8], dec_blocks=[2, 2, 2, 2], mid_blocks=8),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(configs.keys())}")
    return NAFNetSR(**configs[variant])


if __name__ == "__main__":
    # Quick smoke test
    model = build_model("base")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NAFNet-SR (base): {n_params / 1e6:.2f}M parameters")

    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    assert y.shape == (2, 1, 256, 256), "Shape mismatch!"
    print("✓ Forward pass OK")
