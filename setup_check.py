"""
Setup verification script for Whisper Large V3 with CUDA 12.4
Checks all dependencies and hardware configuration
"""
import sys

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current:", sys.version)
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_torch():
    """Check PyTorch installation and CUDA support"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available. GPU acceleration disabled.")
            print("  Make sure you installed PyTorch with CUDA 12.4 support:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_transformers():
    """Check Transformers library"""
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        return True
    except ImportError:
        print("❌ Transformers not installed")
        return False

def check_gradio():
    """Check Gradio"""
    try:
        import gradio
        print(f"✓ Gradio {gradio.__version__}")
        return True
    except ImportError:
        print("❌ Gradio not installed")
        return False

def check_yt_dlp():
    """Check yt-dlp"""
    try:
        import yt_dlp
        print(f"✓ yt-dlp {yt_dlp.version.__version__}")
        return True
    except ImportError:
        print("❌ yt-dlp not installed")
        return False

def check_ffmpeg():
    """Check FFmpeg availability"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ {version_line}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("⚠ FFmpeg not found in PATH")
    print("  Download from: https://ffmpeg.org/download.html")
    return False

def main():
    print("=" * 60)
    print("Whisper Large V3 - Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", check_torch),
        ("Transformers", check_transformers),
        ("Gradio", check_gradio),
        ("yt-dlp", check_yt_dlp),
        ("FFmpeg", check_ffmpeg),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"Checking {name}...")
        result = check_func()
        results.append((name, result))
        print()
    
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All checks passed! You're ready to run the app.")
        print("   Run: python app.py")
    else:
        print("⚠ Some checks failed. Please install missing dependencies.")
        print("   See INSTALLATION.md for detailed instructions.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

