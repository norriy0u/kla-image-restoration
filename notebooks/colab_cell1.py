
# KLA Image Restoration — Colab Training Notebook
# Run this on Google Colab with a GPU runtime (T4 or A100)
# Runtime → Change runtime type → GPU

# ============================================================
# CELL 1: Check GPU
# ============================================================
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "No GPU detected — switch to GPU runtime!")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
