import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import whisper  # type: ignore
from config.settings import WHISPER_MODEL
from transcribe.utils.paths import get_output_paths, validate_output_paths

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    segments: List[Dict[str, Any]]
    language: Optional[str] = None
    duration: Optional[float] = None
    audio_path: Optional[Path] = None


class VideoTranscriber:
    def __init__(self, model_size: str = WHISPER_MODEL, device: Optional[str] = None):
        self.model = whisper.load_model(model_size, device=device)

    async def process_video(
        self,
        video_path: Path,
        overwrite: bool = False,
        **transcribe_kwargs
    ) -> TranscriptionResult:
        """
        Main pipeline: extract audio, transcribe, save results
        """
        component_dir, txt_path, json_path = get_output_paths(video_path)
        audio_path = component_dir / f"{video_path.stem}.wav"

        # 1. Validate paths before processing
        validate_output_paths(txt_path, json_path, overwrite)

        # 2. Extract audio
        await self._extract_audio(video_path, audio_path)

        # 3. Transcribe
        result = await self._transcribe_audio(audio_path, **transcribe_kwargs)

        # 4. Save outputs
        await self._atomic_write(txt_path, result.text)
        await self._atomic_write(json_path, json.dumps(result.segments, indent=2, ensure_ascii=False))

        result.audio_path = audio_path  # Ensure this is stored in result
        return result

    async def _extract_audio(self, video_path: Path, output_path: Path) -> None:
        """
        Extracts audio from video using FFmpeg
        """
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", "-f", "wav",
            str(output_path)
        ]
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with return code {proc.returncode}")

    async def _transcribe_audio(self, audio_path: Path, **kwargs) -> TranscriptionResult:
        """
        Transcribes audio using Whisper and calculates audio duration
        """
        result = await asyncio.to_thread(
            self.model.transcribe,
            str(audio_path),
            **kwargs
        )
        duration = await self._get_audio_duration(audio_path)
        return TranscriptionResult(
            text=result["text"].strip(),
            segments=result.get("segments", []),
            language=result.get("language"),
            duration=duration,
            audio_path=audio_path
        )

    async def _get_audio_duration(self, audio_path: Path) -> float:
        """
        Use ffprobe to get duration of extracted audio
        """
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return float(stdout.decode().strip()) if proc.returncode == 0 else 0.0

    async def _atomic_write(self, path: Path, content: str) -> None:
        """
        Safely writes a file using a temporary path and renaming
        """
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(path)
        except Exception as e:
            if tmp_path.exists():
                await self._safe_remove(tmp_path)
            raise RuntimeError(f"Failed to write file {path}: {str(e)}")

    async def _safe_remove(self, path: Path) -> None:
        """
        Removes file safely, suppressing all errors
        """
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")
