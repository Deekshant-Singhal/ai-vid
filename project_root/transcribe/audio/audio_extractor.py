import os
import subprocess
import tempfile
from typing import Optional

def extract_audio_from_video(
    video_path: str,
    output_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
    max_duration: Optional[float] = None,
    mp3_quality: int = 2
) -> str:
    """
    Extract audio from video with enhanced controls.
    
    Args:
        video_path: Input video file path
        output_format: 'wav' or 'mp3'
        sample_rate: Output sample rate in Hz
        channels: 1 (mono) or 2 (stereo)
        max_duration: Maximum duration in seconds (optional)
        mp3_quality: MP3 quality (0-9, 0=best)
    
    Returns:
        Path to temporary audio file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_format not in ["wav", "mp3"]:
        raise ValueError("Output format must be 'wav' or 'mp3'")
    
    if channels not in [1, 2]:
        raise ValueError("Channels must be 1 (mono) or 2 (stereo)")
    
    if output_format == "mp3" and not (0 <= mp3_quality <= 9):
        raise ValueError("MP3 quality must be between 0 (best) and 9 (worst)")

    suffix = f".{output_format}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
        command = _get_ffmpeg_command(
            video_path, 
            audio_file.name, 
            output_format,
            sample_rate,
            channels,
            mp3_quality
        )
        
        if max_duration:
            command.extend(["-t", str(max_duration)])

        try:
            subprocess.run(
                command, 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out after 30 seconds")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode().strip()
            if "Invalid data found" in error_msg:
                raise ValueError("Invalid video file format") from e
            raise RuntimeError(f"Audio extraction failed: {error_msg}") from e

        if os.path.getsize(audio_file.name) == 0:
            raise RuntimeError("FFmpeg created an empty audio file")

        return audio_file.name

def _get_ffmpeg_command(
    video_path: str,
    output_path: str,
    fmt: str,
    sample_rate: int,
    channels: int,
    mp3_quality: int
) -> list:
    base_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate)
    ]
    
    if fmt == "wav":
        return base_cmd + [
            "-acodec", "pcm_s16le",
            "-f", "wav",
            output_path
        ]
    elif fmt == "mp3":
        return base_cmd + [
            "-acodec", "libmp3lame",
            "-q:a", str(mp3_quality),
            "-f", "mp3",
            output_path
        ]