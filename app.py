import torch
import gradio as gr
# import yt_dlp as youtube_dl
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
from src.run_server import RunServer, NgrokServer
from src.youtube_download_and_transcribe import TranscribeAndTranslate

cuda_and_gpu_check = CudaAndGPUCheck()
pipe = cuda_and_gpu_check.asr_pipeline()
device_config = cuda_and_gpu_check.get_device()
run_server = NgrokServer()
yt_dl_and_trans = TranscribeAndTranslate()

MODEL_NAME = Config.MODEL_NAME
BATCH_SIZE = Config.BATCH_SIZE
FILE_LIMIT_MB = Config.FILE_LIMIT_MB
YT_LENGTH_LIMIT_S = Config.YT_LENGTH_LIMIT_S

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
            with gr.Column():
                language = gr.Dropdown(
                    ["None", "Spanish", "French", "German", "Chinese", "Portuguese", "Russian"],
                    value="None",
                    label="Select Destination Language for Translation",
                    info="Only appears when 'translate' is selected.",
                    interactive=True
                )
            
        yt_submit_btn = gr.Button("🚀 Transcribe YouTube Video", variant="primary")
        yt_html_output = gr.HTML(label="Video Preview")
        
        with gr.Column():
            yt_text_output = gr.Textbox(
                label="📝 Transcription with Timestamps",
                lines=20,
                placeholder="Transcription with timestamps will appear here...",
                elem_classes=["transcription-output"]
            )
        with gr.Column():
            yt_copy_btn = gr.Button("📋 Copy to Clipboard", variant="secondary")
            yt_copy_status = gr.Textbox(label="Status", visible=False)
            yt_copy_btn.click(
                fn=lambda x: "✓ Copied to clipboard!",
                inputs=yt_text_output,
                outputs=yt_copy_status,
                js="(text) => {if(text) navigator.clipboard.writeText(text);}"
            )
        
        yt_submit_btn.click(
            fn=yt_dl_and_trans.from_youtube,
            show_progress='full',
            inputs=[yt_task_input, yt_url_input, language],
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
            fn=yt_dl_and_trans.from_audio_file,
            inputs=[task_input, audio_input, language],
            outputs=output_text
        )

if __name__ == "__main__":
    # run_server.run_server(demo)
    demo.launch()