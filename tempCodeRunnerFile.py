
        return True
    except ImportError:
        print("❌ Gradio not installed")
        return False

def check_yt_dlp():
    """Check yt-dlp"""