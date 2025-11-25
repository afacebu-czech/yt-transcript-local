# GPU/NPU Setup Guide - FORCED GPU/NPU ONLY MODE

This app is now configured to **STRICTLY use GPU or NPU ONLY**. CPU fallback has been completely removed.

## ✅ SOLUTION FOR RTX 5060

Your RTX 5060 GPU has CUDA compute capability **sm_120** (Blackwell architecture), which requires:
- **PyTorch 2.10.0+ nightly** with **CUDA 12.8+**

## Installation Steps

### Step 1: Uninstall Current PyTorch

```bash
pip uninstall torch torchvision torchaudio -y
```

### Step 2: Install PyTorch Nightly with CUDA 12.8

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Note:** torchvision and torchaudio may not be available for Python 3.13, but torch alone is sufficient for Whisper.

### Step 3: Verify Installation

```bash
python verify_cuda.py
```

You should see:
- ✓ CUDA available: 12.8
- ✓ GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- ✓ GPU Memory: ~8 GB

### Step 4: Test GPU Compatibility

```bash
python -c "import torch; x = torch.ones(1, device=0); print('GPU works!', (x * 2).item())"
```

If this prints "GPU works! 2.0", your GPU is ready!

## What Changed

1. **Removed all CPU fallback logic** - App will fail if GPU/NPU is not available
2. **Added GPU compatibility testing** - Tests GPU at startup before loading model
3. **Clear error messages** - Shows exactly what's wrong and how to fix it
4. **Forced GPU/NPU usage** - No way to accidentally use CPU

## Why CUDA 12.8?

- RTX 5060 (Blackwell architecture, sm_120) requires CUDA 12.8+ for full support
- PyTorch 2.10.0+ nightly includes kernels for sm_120
- CUDA 12.8 is backward compatible with CUDA 12.4 drivers

## Error Messages

If you see errors about "no kernel image available" or "CUDA error", it means:
- Your GPU is detected but PyTorch doesn't have compatible kernels
- **Solution:** Install PyTorch 2.10.0+ nightly with CUDA 12.8 (see above)

## NPU Support

If you have an Intel NPU, the app will automatically detect and use it if GPU is not available.

## Troubleshooting

### "No GPU or NPU detected"
- Check NVIDIA drivers: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`
- Reinstall PyTorch with CUDA 12.8 support

### "GPU test failed"
- Install PyTorch 2.10.0+ nightly with CUDA 12.8 (see above)
- Check GPU compute capability compatibility
- Update NVIDIA drivers to latest version

### App won't start
- The app is designed to fail if GPU/NPU is not available
- This is intentional - no CPU fallback
- Fix GPU/NPU setup before running

## Performance

Once GPU is working:
- **Speed**: ~2-5x faster than CPU
- **Memory**: ~6-8GB VRAM for Whisper Large V3
- **Precision**: float16 for optimal performance

## Current Status

✅ **PyTorch 2.10.0.dev20251124+cu128** - Installed and working
✅ **CUDA 12.8** - Compatible with RTX 5060
✅ **GPU Test** - Passed successfully
