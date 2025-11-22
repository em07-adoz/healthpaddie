# modules/tts.py
from gtts import gTTS
import os

def speak_text(text: str, lang_code: str = "en", filename: str = "healthpaddie_reply.mp3"):
    """
    Simple TTS using gTTS.
    lang_code examples:
      - "en" for English
      - "ha" for Hausa (limited)
      - "yo" for Yoruba (limited)
      - "ig" for Igbo (might fallback / not fully supported)
    """
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(filename)
        print(f"🔊 Audio saved as {filename}. Play it with your media player.")
    except Exception as e:
        print(f"⚠ TTS error: {e}")
