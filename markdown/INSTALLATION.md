# Installation Guide for Whisper Large V3 with CUDA 12.4

This guide will help you set up the Whisper transcription app to use your GPU (CUDA 12.4) and NPU.

## Prerequisites

- Windows 10/11
- CUDA 12.4 installed (already done ✓)
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (or Intel NPU)

## Step 1: Install PyTorch with CUDA 12.4 Support

First, install PyTorch with CUDA 12.4 support. Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select:
- **OS:** Windows
- **Package:** Pip
- **Language:** Python
- **Compute Platform:** CUDA 12.4

Or use this command (for CUDA 12.4):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Step 2: Install Other Dependencies

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install transformers>=4.36.0
pip install accelerate>=0.25.0
pip install yt-dlp>=2023.12.30
pip install gradio>=4.0.0
pip install ffmpeg-python>=0.2.0
```

## Step 3: Verify CUDA Installation

Run the verification script:

```bash
python verify_cuda.py
```

You should see:
- Torch version
- CUDA available: True
- CUDA device: [Your GPU name]

## Step 4: Install FFmpeg (Required for Audio Processing)

FFmpeg is required for processing audio files. Download from:
- https://ffmpeg.org/download.html

Or use chocolatey (if installed):
```bash
choco install ffmpeg
```

Make sure FFmpeg is in your system PATH.

## Step 5: Run the Application

```bash
python app.py
```

The app will:
1. Detect your GPU/NPU automatically
2. Load the Whisper Large V3 model
3. Start a local web server at `http://127.0.0.1:7860`

## Troubleshooting

### GPU Not Detected

1. Verify CUDA is installed: `nvcc --version`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with correct CUDA version

### Out of Memory Errors

- Reduce `BATCH_SIZE` in `app.py` (currently 8)
- Use a smaller Whisper model (e.g., `whisper-medium` or `whisper-small`)

### NPU Support

For Intel NPU support, install:
```bash
pip install intel-extension-for-pytorch
```

## Performance Tips

- The app uses **float16** precision on GPU for faster inference
- GPU memory is automatically cleared between transcriptions
- Chunked processing is enabled for long audio files

## Usage

1. Open the web interface at `http://127.0.0.1:7860`
2. Go to "YouTube Transcription" tab
3. Paste a YouTube URL
4. Click "Transcribe YouTube Video"
5. Wait for transcription (faster on GPU!)

