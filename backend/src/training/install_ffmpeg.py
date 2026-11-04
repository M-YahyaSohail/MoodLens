import shutil


def ensure_ffmpeg():
    """Verify that the FFmpeg binary is available on PATH for local use."""
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        print("FFmpeg not found on PATH.")
        print("Install it so the 'ffmpeg' command is accessible, e.g.:")
        print(
            "  - Windows: winget install Gyan.FFmpeg  (or download from https://ffmpeg.org)"
        )
        print("  - macOS:   brew install ffmpeg")
        print("  - Linux:   sudo apt install ffmpeg")
        return False

    print(f"FFmpeg found: {ffmpeg_path}")
    return True
