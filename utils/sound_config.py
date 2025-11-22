import os
from pathlib import Path

def play_audio(file_path: str):
    """
    Opens an audio file (MP3, WAV, etc.) using the default system player on Windows.
    """
    try:
        # Ensure we have a nice absolute path
        full_path = str(Path(file_path).resolve())
        os.startfile(full_path)  # Windows-only
    except Exception as e:
        print(f"⚠ Error playing audio: {e}")
