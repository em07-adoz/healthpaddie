# modules/tts.py
from gtts import gTTS

import os
from pathlib import Path
from groq import Groq
from utils.sound_config import play_audio
from dotenv import load_dotenv

load_dotenv()


GROG_API_KEY= os.getenv("GROG_API_KEY")


def text_to_speech(text: str, output_filename: str = "speech.mp3") -> str:
    """
    Converts a text string into an MP3 audio file using Groq TTS.

    Args:
        text (str): The text to convert to speech.
        output_filename (str): Output MP3 file name. Defaults to 'speech.mp3'.

    Returns:
        str: The path to the generated MP3 audio file.
    """

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    client = Groq()

    # Save file in same folder as this script
    output_path = Path(__file__).parent / output_filename

    response = client.audio.speech.create(
        model="playai-tts",
        voice="Aaliyah-PlayAI",
        response_format="mp3",
        input=text,
    )

    # Write to file
    response.write_to_file(f"{output_path}")

    return str(output_path)

