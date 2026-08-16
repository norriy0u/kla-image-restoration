import sys, os
sys.path.insert(0, '.')

BASE = os.path.expanduser('~/Downloads/kla_extracted')
GT_DIR = os.path.join(BASE, 'train/train/GT')
LR_DIR = os.path.join(BASE, 'train/train/NoisyLR')
TEST_DIR = os.path.join(BASE, 'test_noisyLR/NoisyLR')

print('=== 1. Dataset ===')
from dataset import RestorationDataset, InferenceDataset
ds_train = RestorationDataset(GT_DIR, LR_DIR, split='train', augment=True)
ds_val   = RestorationDataset(GT_DIR, LR_DIR, split='val',   augment=False)
ds_test  = InferenceDataset(TEST_DIR)

s = ds_train[0]
print('GT shape:', s['gt'].shape, 'range:', round(s['gt'].min().item(),3), round(s['gt'].max().item(),3))
print('LR shape:', s['lr'].shape, 'range:', round(s['lr'].min().item(),3), round(s['lr'].max().item(),3))
t = ds_test[0]
print('Test LR:', t['lr'].shape, 'range:', round(t['lr'].min().item(),3), round(t['lr'].max().item(),3))

print()
print('=== 2. Model ===')
import torch
from model.nafnet import build_model
model = build_model('base')
n = sum(p.numel() for p in model.parameters())
print('NAFNet-SR (base):', round(n/1e6, 2), 'M params')

x = s['lr'].unsqueeze(0)
with torch.no_grad():
    y = model(x)
print('Input:', tuple(x.shape))
print('Output:', tuple(y.shape), 'range:', round(y.min().item(),3), round(y.max().item(),3))
assert y.shape == (1, 1, 256, 256), 'Shape mismatch!'

print()
print('=== 3. Loss ===')
from model.losses import CompositeLoss
loss_fn = CompositeLoss()
gt = s['gt'].unsqueeze(0)
total, breakdown = loss_fn(y, gt)
print('Total loss:', round(total.item(), 4))
for k, v in breakdown.items():
    print(' ', k, round(v, 4))

print()
print('=== 4. Metrics ===')
from utils.metrics import compute_psnr, compute_ssim
print('PSNR (untrained):', round(compute_psnr(y, gt), 2), 'dB')
print('SSIM (untrained):', round(compute_ssim(y, gt), 4))

print()
print('ALL TESTS PASSED')
