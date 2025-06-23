import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
import numpy as np

from transcribe.video_transcriber import VideoTranscriber, TranscriptionResult
from summarization.text.summarizer import Summarizer, SummaryResult, Style
from transcribe.utils.paths import get_output_paths, validate_output_paths
from summarization.audio.audio_analyzer import AudioAnalyzer, AudioSegment

logger = logging.getLogger(__name__)

class VideoSummaryPipeline:
    def __init__(
        self,
        *,
        transcriber: Optional[VideoTranscriber] = None,
        summarizer: Optional[Summarizer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        enable_audio_fusion: bool = True
    ):
        self.transcriber = transcriber or VideoTranscriber()
        self.summarizer = summarizer or Summarizer()
        self.audio_analyzer = audio_analyzer or AudioAnalyzer(noise_reduce=True)
        self.enable_audio_fusion = enable_audio_fusion

    async def process_video(
        self,
        video_path: Path,
        style: str = "default",
        max_highlights: int = 5,
        overwrite: bool = False,
        **transcribe_kwargs
    ) -> Dict[str, Any]:
        try:
            # Get paths and validate
            component_dir, txt_path, json_path = get_output_paths(video_path)
            summary_path = component_dir / f"{video_path.stem}_summary.json"

            validate_output_paths(txt_path, json_path, overwrite)
            if not overwrite and summary_path.exists():
                raise FileExistsError(f"Summary file exists: {summary_path}")

            # Step 1: Transcribe video (with retries)
            transcription = await self._retry_operation(
                self.transcriber.process_video,
                video_path,
                overwrite=overwrite,
                **transcribe_kwargs
            )

            # Parallel execution of text and audio analysis
            summary_task = asyncio.create_task(
                self._summarize_transcript(
                    transcription.text,
                    style,
                    max_highlights
                )
            )
            
            audio_analysis = None
            if self.enable_audio_fusion:
                audio_task = asyncio.create_task(
                    self._analyze_audio(
                        transcription.audio_path,
                        transcription.segments
                    )
                )
                audio_analysis = await audio_task

            summary = await summary_task

            # Fuse audio and text features
            if audio_analysis:
                summary = self._fuse_audio_features(summary, audio_analysis)

            # Save results
            await self._atomic_write(
                summary_path,
                self._format_output(
                    transcription,
                    summary,
                    str(component_dir)
                )
            )

            return self._format_output(
                transcription,
                summary,
                str(component_dir)
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed for {video_path}: {str(e)}", exc_info=True)
            raise PipelineError(f"Video processing failed: {str(e)}") from e

    async def _retry_operation(self, func, *args, **kwargs):
        """Retry wrapper with exponential backoff"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s")
                await asyncio.sleep(delay)

    async def _summarize_transcript(
        self,
        text: str,
        style: str,
        max_highlights: int
    ) -> SummaryResult:
        """Wrapper for summarization with error context"""
        try:
            return self.summarizer.summarize(
                transcript=text,
                style=style,
                max_highlights=max_highlights
            )
        except Exception as e:
            logger.error("Summarization failed", exc_info=True)
            raise PipelineError("Text summarization failed") from e

    async def _analyze_audio(
        self,
        audio_path: Path,
        segments: List[Dict]
    ) -> Dict[tuple, Dict[str, Any]]:
        """Audio analysis with feature extraction"""
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file missing: {audio_path}")
                
            segments = self.audio_analyzer.batch_analyze(audio_path, segments)
            return {
                (round(s.start, 2), round(s.end, 2)): {
                    'confidence': s.confidence,
                    'emotion': s.emotion,
                    'speech_prob': s.speech_prob,
                    'label': s.label
                }
                for s in segments
            }
        except Exception as e:
            logger.error("Audio analysis failed", exc_info=True)
            raise PipelineError("Audio processing failed") from e

    def _fuse_audio_features(self, summary: SummaryResult, audio_analysis: Dict) -> SummaryResult:
        """Enhance text highlights with audio features"""
        for highlight in summary.highlights:
            if isinstance(highlight.start, list):
                # Multi-segment highlight
                audio_data = [
                    audio_analysis.get((round(s, 2), round(e, 2)), {})
                    for s, e in zip(highlight.start, highlight.end)
                ]
                highlight.audio_confidence = np.mean([d.get('confidence', 0) for d in audio_data])
                
                # Corrected emotion detection
                emotion_pairs = [(d.get('emotion', 'neutral'), d.get('confidence', 0)) 
                            for d in audio_data]
                highlight.emotion = max(emotion_pairs, key=lambda x: x[1])[0]
                
            else:
                # Single segment highlight
                key = (round(highlight.start, 2), round(highlight.end, 2))
                audio_data = audio_analysis.get(key, {})
                highlight.audio_confidence = audio_data.get('confidence', 0)
                highlight.emotion = audio_data.get('emotion', 'neutral')
            
            # Adjust final confidence
            if hasattr(highlight, 'audio_confidence'):
                highlight.confidence = (
                    0.7 * highlight.confidence + 
                    0.3 * highlight.audio_confidence
                )
        
        return summary

    def _format_output(
        self,
        transcription: TranscriptionResult,
        summary: SummaryResult,
        component_dir: str
    ) -> Dict[str, Any]:
        """Standardized output format with audio-text fusion"""
        output = {
            "metadata": {
                "pipeline_version": "2.0",
                "processing_time": datetime.now().isoformat(),
                "audio_analysis_enabled": self.enable_audio_fusion
            },
            "transcription": {
                "text": transcription.text,
                "language": transcription.language,
                "duration": transcription.duration,
                "word_count": len(transcription.text.split()),
                "segment_count": len(transcription.segments),
                "audio_path": str(transcription.audio_path) if transcription.audio_path else None
            },
            "summary": {
                "style": summary.style if hasattr(summary, 'style') else "default",
                "overview": summary.overview,
                "keywords": summary.keywords,
                "highlights": []
            },
            "paths": {
                "component_dir": component_dir,
                "transcript": str(Path(component_dir) / f"{Path(transcription.audio_path).stem}.txt"),
                "segments": str(Path(component_dir) / f"{Path(transcription.audio_path).stem}_segments.json"),
                "summary": str(Path(component_dir) / f"{Path(transcription.audio_path).stem}_summary.json")
            }
        }

        # Format highlights with audio metadata
        for h in summary.highlights:
            highlight_data = {
                "text": h.highlight_text,
                "timestamps": self._format_timestamps(h),
                "category": h.category,
                "confidence": round(h.confidence, 3),
                "text_confidence": getattr(h, 'text_confidence', None),
                "audio_metadata": {
                    "confidence": getattr(h, 'audio_confidence', None),
                    "emotion": getattr(h, 'emotion', None),
                    "speech_probability": getattr(h, 'speech_prob', None)
                } if self.enable_audio_fusion else None
            }
            output["summary"]["highlights"].append(highlight_data)

        return output

    def _format_timestamps(self, highlight) -> List[Dict]:
        if isinstance(highlight.start, list):
            return [
                {"start": round(s, 2), "end": round(e, 2)}
                for s, e in zip(highlight.start, highlight.end)
            ]
        return [{"start": round(highlight.start, 2), "end": round(highlight.end, 2)}]

    async def _atomic_write(self, path: Path, data: Dict) -> None:
        """Atomic JSON write with tmp file"""
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            tmp_path.replace(path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise PipelineError(f"Failed to write {path}: {str(e)}")


class PipelineError(Exception):
    """Custom exception for pipeline failures"""
    pass