import os
import subprocess
import tempfile
import whisper
from config import WHISPER_MODEL

def extract_audio(video_path):
    """Extract audio from video using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
        command = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", audio_file.name
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_file.name

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path)
    return result["text"], result["segments"]

def get_transcript_from_video(video_path):
    """Full pipeline: extract + transcribe"""
    audio_path = extract_audio(video_path)
    text, segments = transcribe_audio(audio_path)
    os.remove(audio_path)
    return text, segments
