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

class RunServer:
    def __init__(self):
        self.device_config = CudaAndGPUCheck().get_device() 
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n" + "="*50)
        print("Shutting down gracefully...")
        print("="*50)
        
        # Clear GPU cache if using GPU
        if self.device_config["device_type"] == "CUDA GPU":
            torch.cuda.empty_cache()
            print("✓ GPU cache cleared")
        
        print("✓ Server stopped")
        sys.exit(0)
    
    def run_server(self, app):
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
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

            app.launch(
                server_name="0.0.0.0",  # Allow connections from any network interface
                server_port=7860,
                share=False,  # Set to True for public Gradio link (requires internet)
                show_error=True,
                quiet=False,
            )
        except KeyboardInterrupt:
            self.signal_handler(None, None)
        except Exception as e:
                print(f"\n❌ Error starting server: {e}")
                sys.exit(1)
    
    
    