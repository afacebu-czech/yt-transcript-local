import torch
import numpy as np
from transformers import pipeline

from .config import Config

class CudaAndGPUCheck:
    def __init__(self, device_id: int=0):
        self.device_id = device_id
        self.device_config = self.get_device()
        self.global_config = Config()
        self.pipeline = pipeline
    
    # Device configuration - STRICTLY GPU or NPU ONLY (NO CPU FALLBACK)
    def test_cuda_device(self):
        """Test if CUDA device is actually usable (not just detected)"""
        try:
            # Try a simple operation on the GPU
            test_tensor = torch.ones(1, device=self.device_id)
            result = test_tensor * 2
            del test_tensor, result
            torch.cuda.synchronize(self.device_id)
            return True, None
        except (RuntimeError, Exception) as e:
            return False, str(e)
        
    def get_device(self):
        """Force GPU or NPU ONLY - NO CPU FALLBACK"""
        device_info = None
        
        # PRIORITY 1: Check for CUDA GPU (primary)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            print(f"✓ CUDA GPU detected: {device_name}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Test if GPU actually works
            print("  Testing GPU compatibility...")
            gpu_works, error_msg = self.test_cuda_device()
            
            if gpu_works:
                device_info = {
                    "device": 0,
                    "torch_dtype": torch.float16,
                    "device_name": device_name,
                    "device_type": "CUDA GPU"
                }
                print(f"  ✓ GPU is usable - FORCING GPU acceleration")
            else:
                print(f"  ❌ GPU test failed: {error_msg[:150] if error_msg else 'Unknown error'}")
                print(f"\n" + "="*70)
                print("❌ GPU NOT COMPATIBLE - APP REQUIRES GPU/NPU")
                print("="*70)
                print(f"\nYour GPU ({device_name}) is detected but not compatible with current PyTorch.")
                print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]} (sm_{compute_capability[0]}{compute_capability[1]})")
                print(f"\n🔧 SOLUTION: Install PyTorch Nightly with CUDA 12.4 support:")
                print(f"   pip uninstall torch torchvision torchaudio")
                print(f"   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                print(f"\nOr wait for official PyTorch release with sm_{compute_capability[0]}{compute_capability[1]} support.")
                print("="*70 + "\n")
                raise RuntimeError(
                    f"GPU ({device_name}) is not compatible with current PyTorch build. "
                    f"Install PyTorch nightly for sm_{compute_capability[0]}{compute_capability[1]} support. "
                    f"Error: {error_msg[:200] if error_msg else 'GPU kernel not available'}"
                )
        
        # PRIORITY 2: Check for Intel NPU (if GPU not available)
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_info = {
                "device": "xpu:0",
                "torch_dtype": torch.float16,
                "device_name": "Intel NPU",
                "device_type": "Intel NPU"
            }
            print(f"✓ Using Intel NPU (GPU not available)")
        
        # PRIORITY 3: Check for other NPU backends
        elif hasattr(torch.backends, 'qnnpack'):
            # Add specific NPU detection here if needed
            pass
        
        # NO GPU OR NPU FOUND - FAIL
        if device_info is None:
            print("\n" + "="*70)
            print("❌ NO GPU OR NPU DETECTED - APP REQUIRES GPU/NPU")
            print("="*70)
            print("\nThis app is configured to use GPU or NPU ONLY.")
            print("CPU fallback is disabled.")
            print("\nPlease ensure:")
            print("  1. NVIDIA GPU is installed and drivers are up to date")
            print("  2. CUDA 12.4 is properly installed")
            print("  3. PyTorch with CUDA support is installed:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("="*70 + "\n")
            raise RuntimeError("No GPU or NPU detected. This app requires GPU/NPU acceleration and does not support CPU.")
        
        return device_info
    
    def asr_pipeline(self):
        pipe = self.pipeline(
            task="automatic-speech-recognition",
            model=self.global_config.MODEL_NAME,
            device=self.device_config["device"],
            dtype=self.device_config["torch_dtype"],
            model_kwargs={
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            },
            ignore_warning=True,  # Suppress chunk_length_s warning since we're using Whisper's native chunking
        )
        return pipe
    
    def initialize_pipeline(self):
        # Initialize pipeline - GPU/NPU ONLY, NO CPU FALLBACK
        try:
            # Initialize pipeline without chunk_length_s for better long-form transcription
            # Whisper handles its own chunking mechanism internally (see Whisper paper section 3.8)
            pipe = self.asr_pipeline
            
            # Test the pipeline with a dummy operation to ensure it works
            print("  Testing pipeline with device...")
            test_audio = {"array": np.array([0.0] * 16000, dtype=np.float32), "sampling_rate": 16000}
            _ = pipe(test_audio, return_timestamps=False)
            
            if self.device_config["device_type"] == "CUDA GPU":
                print("✓ GPU pipeline initialized and tested successfully")
                torch.cuda.empty_cache()  # Clear cache for optimal memory usage
                # Enable optimizations
                if hasattr(pipe.model, "half"):
                    pipe.model = pipe.model.half()
            else:
                print(f"✓ {self.device_config['device_type']} pipeline initialized successfully")
                
        except RuntimeError as e:
            error_str = str(e)
            if "CUDA" in error_str or "cuda" in error_str.lower() or "kernel" in error_str.lower():
                print(f"\n" + "="*70)
                print("❌ GPU PIPELINE INITIALIZATION FAILED")
                print("="*70)
                print(f"\nError: {error_str[:300]}")
                print(f"\nYour GPU ({self.device_config.get('device_name', 'Unknown')}) is not compatible.")
                print(f"\n🔧 SOLUTION: Install PyTorch Nightly:")
                print(f"   pip uninstall torch torchvision torchaudio")
                print(f"   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                print("="*70 + "\n")
                raise RuntimeError(
                    f"GPU pipeline initialization failed. GPU is not compatible with current PyTorch. "
                    f"Install PyTorch nightly for support. Error: {error_str[:200]}"
                ) from e
            else:
                raise RuntimeError(f"Pipeline initialization failed: {error_str[:200]}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline: {str(e)[:200]}") from e
        
    def print_pipeline_status(self):
        # Initialize pipeline with GPU/NPU optimizations
        print(f"\nLoading Whisper model: {self.global_config.MODEL_NAME}")
        print(f"Device: {self.device_config['device_type']} ({self.device_config['device_name']})")
        print(f"Data Type: {self.device_config['torch_dtype']}")
        
    
def main():
    cagc = CudaAndGPUCheck()
    # Get device configuration - FORCES GPU/NPU ONLY
    cagc.print_pipeline_status()
    
if __name__ == "__main__":
    main()