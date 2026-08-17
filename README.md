# KLA Semiconductor Image Restoration — ReDI-NAFNet

AI-based restoration of degraded semiconductor microscopic inspection images for **SEMICON India Hackathon 2026**.
Handles speckle noise, Gaussian noise, and 2× super-resolution in a single forward pass.

- **Model Variant**: ReDI-NAFNet Tiny (**1.81M Parameters**) — Nonlinear Activation Free Network + Sub-Pixel PixelShuffle SR head
- **Input**: 128×128 NoisyLR `.npy` arrays (float32, handles speckle noise overflow >1.0+)
- **Output**: 256×256 Restored `.npy` arrays (float32, range [0,1])

---

## 📊 Benchmark Results

Evaluated across **3,200 dataset images**:

| Metric | Score | Target / Description |
|---|---|---|
| **PSNR** | **25.49 dB** | Peak Signal-to-Noise Ratio (Baseline Checkpoint) |
| **SSIM** | **0.6669** | Structural Similarity Index Measure |
| **LPIPS** | **0.4387** | Learned Perceptual Quality Score |
| **Inference Latency** | **36.5 ms / image** | Self-measured CPU Baseline (<5ms projected on H100 GPU) |
| **Throughput** | **27.4 images / sec** | Edge manufacturing pipeline compatible |

---

## ⚡ Quick Start — Standalone Inference

```bash
git clone https://github.com/norriy0u/kla-image-restoration.git
cd kla-image-restoration
pip install -r requirements.txt

# Run standalone evaluation / test inference
python evaluate.py \
    --input_dir /path/to/test_noisyLR/NoisyLR \
    --output_dir ./outputs/test \
    --variant tiny
```

Restored images are saved as `.npy` (float32) and `.png` (8-bit) in `./outputs/test/`.

---

## 📁 Repository Structure

```
kla-image-restoration/
├── evaluate.py          ← CRITICAL: standalone inference script for KLA benchmarking
├── train.py             ← complete model training script with auto-resume
├── dataset.py           ← paired dataset loader (.npy format & speckle normalization)
├── fill_ppt.py          ← automated PowerPoint presentation generator
├── generate_comparison_plot.py ← 4-panel visual interpretability generator
├── model/
│   ├── nafnet.py        ← NAFNet-SR architecture (SimpleGate + PixelShuffle SR)
│   └── losses.py        ← composite loss (Charbonnier + FFT freq + SSIM + Sobel edge)
├── utils/
│   └── metrics.py       ← PSNR, SSIM, LPIPS metrics calculation
├── weights/
│   └── best_model.pt    ← trained 1.81M param model weights checkpoint (21.95 MB)
└── requirements.txt
```

---

## 📊 Presentation Generator (`fill_ppt.py`)

To populate the official hackathon PowerPoint template automatically:
1. Make sure `Idea-Submission-Template_Hackathon-2026-1.pptx` is in your `Downloads` folder.
2. Edit your team details in `fill_ppt.py`.
3. Run:
   ```bash
   python fill_ppt.py
   ```
4. Output presentation will be saved to your `Downloads` folder as `KLA_Restoration_Final_Submission.pptx`.

---

## 🔬 Model Architecture & Loss Function

1. **NAFNet Backbone**: Removes non-linear activations (GELU/ReLU) to avoid low-level feature degradation.
2. **SimpleGate & SCA**: Dynamic channel attention feature routing for multi-degradation adaptation.
3. **PixelShuffle Sub-Pixel SR**: 2× spatial resolution upsampling ($128\times128 \rightarrow 256\times256$).
4. **Composite Loss**:
   $$\mathcal{L}_{\text{total}} = 1.0\mathcal{L}_{\text{Charbonnier}} + 0.1\mathcal{L}_{\text{FFT}} + 0.2\mathcal{L}_{\text{SSIM}} + 0.1\mathcal{L}_{\text{Sobel}}$$

---

## 📜 License & Citation

Developed for **SEMICON India Hackathon 2026 — KLA Semiconductor Problem Statement**.
- NAFNet: Chen et al., *"Simple Baselines for Image Restoration"*, ECCV 2022.
- Spectral Loss: Cho et al., *"Rethinking Coarse-to-Fine Approach in Single Image Deblurring"*, CVPR 2021.
