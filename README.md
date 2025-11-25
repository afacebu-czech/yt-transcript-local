---
title: Whisper Large V3 - YouTube Transcriber
emoji: 🎙️
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.46.0
app_file: app.py
pinned: false
---

# 🎙️ Whisper Large V3 - YouTube Transcriber

A local transcription app using OpenAI's Whisper Large V3 model, optimized for GPU (CUDA 12.4) and NPU acceleration.

## Features

- 🚀 **GPU/NPU Accelerated**: Automatically uses CUDA GPU or NPU for fast transcription
- 📺 **YouTube Support**: Paste any YouTube URL to transcribe
- 🎵 **Audio File Support**: Upload and transcribe local audio files
- 🌍 **Multilingual**: Supports 99 languages
- 🔄 **Translation**: Translate speech to English
- ⚡ **Optimized**: Uses float16 precision on GPU for maximum performance

## Quick Start

### 1. Verify Setup

```bash
python setup_check.py
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

The app will start at `http://127.0.0.1:7860`

## Requirements

- Python 3.8+
- CUDA 12.4 (already installed ✓)
- NVIDIA GPU with CUDA support (or Intel NPU)
- FFmpeg (for audio processing)

## Usage

1. Open the web interface
2. Go to "YouTube Transcription" tab
3. Paste a YouTube URL
4. Select task (transcribe or translate)
5. Click "Transcribe YouTube Video"

## Model

- **Model**: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- **Source**: Based on [Hugging Face Whisper Space](https://huggingface.co/spaces/openai/whisper)

## Performance

- **GPU**: ~2-5x faster than CPU
- **Memory**: ~6-8GB VRAM for Whisper Large V3
- **Speed**: ~1-2 minutes per hour of audio on GPU

## Troubleshooting

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions and troubleshooting.

## License

Apache 2.0 (same as Whisper model)
