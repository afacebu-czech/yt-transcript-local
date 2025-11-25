import torch
import gradio as gr
import yt_dlp as youtube_dl
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import numpy as np

import tempfile
import os
import time
import signal
import sys

from src.utils.config import Config
from src.utils.cuda_and_gpu_check import CudaAndGPUCheck

cuda_and_gpu_check = CudaAndGPUCheck()
pipe = cuda_and_gpu_check.asr_pipeline()
device_config = cuda_and_gpu_check.get_device()

MODEL_NAME = Config.MODEL_NAME
BATCH_SIZE = Config.BATCH_SIZE
FILE_LIMIT_MB = Config.FILE_LIMIT_MB
YT_LENGTH_LIMIT_S = Config.YT_LENGTH_LIMIT_S

def format_timestamp(seconds):
    """Format seconds to HH:MM:SS or MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def format_transcription_with_timestamps(result, word_level=False):
    """Format transcription with timestamps for better readability"""
    # Handle different result formats from Whisper
    if isinstance(result, dict):
        if "chunks" in result:
            chunks = result["chunks"]
        elif "text" in result and "chunks" not in result:
            # Simple text result, try to get chunks from the full result
            if hasattr(result, "chunks"):
                chunks = result.chunks
            else:
                # Fallback: return text with a note
                return f"{result.get('text', '')}\n\n(Note: Timestamps not available in this format)"
        else:
            return str(result.get("text", str(result)))
    else:
        return str(result)
    
    if not chunks:
        return "No transcription chunks found."
    
    formatted_lines = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "").strip()
            if not text:
                continue
                
            timestamp = chunk.get("timestamp", None)
            
            if timestamp:
                if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                    start_time = format_timestamp(timestamp[0])
                    end_time = format_timestamp(timestamp[1])
                    formatted_lines.append(f"[{start_time} → {end_time}]\n{text}")
                elif isinstance(timestamp, (list, tuple)) and len(timestamp) == 1:
                    time_str = format_timestamp(timestamp[0])
                    formatted_lines.append(f"[{time_str}]\n{text}")
                elif isinstance(timestamp, (int, float)):
                    time_str = format_timestamp(timestamp)
                    formatted_lines.append(f"[{time_str}]\n{text}")
                else:
                    formatted_lines.append(text)
            else:
                formatted_lines.append(text)
        elif isinstance(chunk, str):
            formatted_lines.append(chunk)
        else:
            formatted_lines.append(str(chunk))
    
    return "\n\n".join(formatted_lines)

def transcribe(inputs, task):
    """Transcribe audio with improved accuracy and timestamp formatting"""
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    try:
        # Improved generation parameters for better accuracy
        generate_kwargs = {
            "task": task,
            "language": None,  # Auto-detect language
            "num_beams": 5,  # Beam search for better accuracy
            "temperature": 0.0,  # Deterministic output
            "compression_ratio_threshold": 2.4,  # Filter out repetitive text
            "logprob_threshold": -1.0,  # Filter low-confidence predictions
            "no_speech_threshold": 0.6,  # Better silence detection
            "condition_on_prev_tokens": True,  # Use context for better accuracy
        }
        
        # Get sentence-level timestamps for better readability
        result = pipeline(
            inputs,
            batch_size=BATCH_SIZE,
            generate_kwargs=generate_kwargs,
            return_timestamps="sentence",  # Sentence-level timestamps
        )
        
        # Format the output with timestamps
        formatted_text = format_transcription_with_timestamps(result, word_level=False)
        
        return formatted_text
    except RuntimeError as e:
        error_str = str(e)
        if "CUDA" in error_str or "cuda" in error_str.lower() or "kernel" in error_str.lower():
            raise gr.Error(
                f"❌ GPU ERROR: {error_str[:300]}\n\n"
                f"This app requires GPU/NPU and does not support CPU fallback.\n"
                f"Your GPU ({device_config.get('device_name', 'Unknown')}) may not be compatible with current PyTorch.\n\n"
                f"🔧 SOLUTION: Install PyTorch Nightly:\n"
                f"   pip uninstall torch torchvision torchaudio\n"
                f"   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124"
            )
        else:
            raise gr.Error(f"Transcription error: {error_str[:200]}")


def _return_yt_html_embed(yt_url):
    """Extract video ID and return HTML embed string"""
    # Handle various YouTube URL formats
    video_id = yt_url
    if "?v=" in yt_url:
        video_id = yt_url.split("?v=")[-1].split("&")[0]
    elif "youtu.be/" in yt_url:
        video_id = yt_url.split("youtu.be/")[-1].split("?")[0]
    elif "/watch/" in yt_url:
        video_id = yt_url.split("/watch/")[-1].split("?")[0]
    
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def download_yt_audio(yt_url, filename):
    """Download audio from YouTube URL"""
    info_loader = youtube_dl.YoutubeDL()
    
    try:
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as err:
        raise gr.Error(f"YouTube download error: {str(err)}")
    
    # Parse duration
    file_length = info.get("duration_string", "0:00")
    file_h_m_s = file_length.split(":")
    file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
    
    if len(file_h_m_s) == 1:
        file_h_m_s.insert(0, 0)
    if len(file_h_m_s) == 2:
        file_h_m_s.insert(0, 0)
    file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
    
    if file_length_s > YT_LENGTH_LIMIT_S:
        yt_length_limit_hms = time.strftime("%H:%M:%S", time.gmtime(YT_LENGTH_LIMIT_S))
        file_length_hms = time.strftime("%H:%M:%S", time.gmtime(file_length_s))
        raise gr.Error(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")
    
    # Download audio only for faster processing
    # Use base filename without extension, yt-dlp will add appropriate extension
    base_filename = os.path.splitext(filename)[0]
    ydl_opts = {
        "outtmpl": base_filename + ".%(ext)s",
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
            # yt-dlp will create a .wav file
            wav_filename = base_filename + ".wav"
            if os.path.exists(wav_filename):
                return wav_filename
            # Fallback: check for original filename
            if os.path.exists(filename):
                return filename
            raise gr.Error("Downloaded file not found")
        except youtube_dl.utils.ExtractorError as err:
            raise gr.Error(f"YouTube extraction error: {str(err)}")


def yt_transcribe(yt_url, task, max_filesize=75.0):
    """Transcribe YouTube video"""
    if not yt_url or not yt_url.strip():
        raise gr.Error("Please provide a valid YouTube URL!")
    
    html_embed_str = _return_yt_html_embed(yt_url)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        filepath = download_yt_audio(yt_url, filepath)
        
        with open(filepath, "rb") as f:
            inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    # Clear GPU cache before processing
    if device_config["device_type"] == "CUDA GPU":
        torch.cuda.empty_cache()
    
    try:
        # Improved generation parameters for better accuracy
        generate_kwargs = {
            "task": task,
            "language": None,  # Auto-detect language
            "num_beams": 5,  # Beam search for better accuracy
            "temperature": 0.0,  # Deterministic output
            "compression_ratio_threshold": 2.4,  # Filter out repetitive text
            "logprob_threshold": -1.0,  # Filter low-confidence predictions
            "no_speech_threshold": 0.6,  # Better silence detection
            "condition_on_prev_tokens": True,  # Use context for better accuracy
        }
        
        # Get word-level timestamps for better precision
        result = pipe(
            inputs,
            batch_size=BATCH_SIZE,
            generate_kwargs=generate_kwargs,
            return_timestamps="sentence",  # Sentence-level timestamps
        )
        
        # Format the output with timestamps
        formatted_text = format_transcription_with_timestamps(result, word_level=False)
        
    except RuntimeError as e:
        error_str = str(e)
        if "CUDA" in error_str or "cuda" in error_str.lower() or "kernel" in error_str.lower():
            raise gr.Error(
                f"❌ GPU ERROR: {error_str[:300]}\n\n"
                f"This app requires GPU/NPU and does not support CPU fallback.\n"
                f"Your GPU ({device_config.get('device_name', 'Unknown')}) may not be compatible with current PyTorch.\n\n"
                f"🔧 SOLUTION: Install PyTorch Nightly:\n"
                f"   pip uninstall torch torchvision torchaudio\n"
                f"   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124"
            )
        else:
            raise gr.Error(f"Transcription error: {error_str[:200]}")

    return html_embed_str, formatted_text


# Create Gradio interface
device_status = f"""
**Device:** {device_config['device_type']}  
**Device Name:** {device_config['device_name']}  
**Data Type:** {device_config['torch_dtype']}  
**Model:** {MODEL_NAME}
"""

with gr.Blocks(title="Whisper Large V3 - YouTube Transcriber") as demo:
    # Custom CSS for better readability (injected via HTML)
    gr.HTML("""
    <style>
    .transcription-output textarea {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        padding: 15px !important;
    }
    </style>
    """, visible=False)
    
    gr.Markdown("# 🎙️ Whisper Large V3: YouTube Transcriber")
    gr.Markdown(
        "Transcribe YouTube videos with OpenAI Whisper Large V3. "
        f"Using [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers. "
        "**Improved accuracy with beam search and timestamp formatting.**"
    )
    gr.Markdown(device_status)
    
    with gr.Tab("📺 YouTube Transcription"):
        with gr.Row():
            yt_url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="Paste YouTube video URL here (e.g., https://www.youtube.com/watch?v=...)",
                lines=1
            )
            yt_task_input = gr.Radio(
                ["transcribe", "translate"],
                label="Task",
                value="transcribe",
                info="Transcribe: same language | Translate: to English"
            )
        yt_submit_btn = gr.Button("🚀 Transcribe YouTube Video", variant="primary")
        yt_html_output = gr.HTML(label="Video Preview")
        
        with gr.Row():
            yt_text_output = gr.Textbox(
                label="📝 Transcription with Timestamps",
                lines=20,
                placeholder="Transcription with timestamps will appear here...",
                elem_classes=["transcription-output"]
            )
        with gr.Row():
            yt_copy_btn = gr.Button("📋 Copy to Clipboard", variant="secondary")
            yt_copy_status = gr.Textbox(label="Status", visible=False)
            yt_copy_btn.click(
                fn=lambda x: "✓ Copied to clipboard!",
                inputs=yt_text_output,
                outputs=yt_copy_status,
                js="(text) => {if(text) navigator.clipboard.writeText(text);}"
            )
        
        yt_submit_btn.click(
            fn=yt_transcribe,
            inputs=[yt_url_input, yt_task_input],
            outputs=[yt_html_output, yt_text_output]
        )
    
    with gr.Tab("🎵 Audio File"):
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            task_input = gr.Radio(
                ["transcribe", "translate"],
                label="Task",
                value="transcribe",
                info="Transcribe: same language | Translate: to English"
            )
        submit_btn = gr.Button("🚀 Transcribe Audio", variant="primary")
        
        with gr.Row():
            output_text = gr.Textbox(
                label="📝 Transcription with Timestamps",
                lines=20,
                placeholder="Transcription with timestamps will appear here...",
                elem_classes=["transcription-output"]
            )
        with gr.Row():
            copy_btn = gr.Button("📋 Copy to Clipboard", variant="secondary")
            copy_status = gr.Textbox(label="Status", visible=False)
            copy_btn.click(
                fn=lambda x: "✓ Copied to clipboard!",
                inputs=output_text,
                outputs=copy_status,
                js="(text) => {if(text) navigator.clipboard.writeText(text);}"
            )
        
        submit_btn.click(
            fn=transcribe,
            inputs=[audio_input, task_input],
            outputs=output_text
        )

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n" + "="*50)
    print("Shutting down gracefully...")
    print("="*50)
    
    # Clear GPU cache if using GPU
    if device_config["device_type"] == "CUDA GPU":
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")
    
    print("✓ Server stopped")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "="*50)
    print("Starting Whisper Transcription Server...")
    print("="*50)
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    try:
        # Get local IP address for network access
        import socket
        def get_local_ip():
            """Get the local IP address"""
            try:
                # Connect to a remote address to determine local IP
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except Exception:
                return "127.0.0.1"
        
        local_ip = get_local_ip()
        
        print(f"\n" + "="*70)
        print("🌐 Server Access Information")
        print("="*70)
        print(f"Local access:    http://127.0.0.1:7860")
        print(f"Network access:  http://{local_ip}:7860")
        print(f"\nOthers on your network can access the app at:")
        print(f"  → http://{local_ip}:7860")
        print(f"\n⚠ Make sure Windows Firewall allows connections on port 7860")
        print("="*70 + "\n")
        
        demo.launch(
            server_name="0.0.0.0",  # Allow connections from any network interface
            server_port=7860,
            share=False,  # Set to True for public Gradio link (requires internet)
            show_error=True,
            quiet=False,
        )
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)
