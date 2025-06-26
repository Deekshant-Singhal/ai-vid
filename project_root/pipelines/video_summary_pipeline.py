import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
import numpy as np

from transcribe.video_transcriber import VideoTranscriber, TranscriptionResult
from summarization.text.summarizer import Summarizer, SummaryResult
from paths import get_output_paths, validate_output_paths
from summarization.audio.audio_analyzer import AudioAnalyzer

logger = logging.getLogger(__name__)

class VideoSummaryPipeline:
    """End-to-end pipeline for video summarization with audio-text fusion"""
    
    def __init__(
        self,
        *,
        transcriber: Optional[VideoTranscriber] = None,
        summarizer: Optional[Summarizer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        enable_audio_fusion: bool = True,
        audio_weight: float = 0.3,
        strict_audio: bool = False,
        max_retries: int = 3
    ):
        """
        Initialize pipeline components
        
        Args:
            transcriber: Video transcription component
            summarizer: Text summarization component
            audio_analyzer: Audio feature extraction component
            enable_audio_fusion: Whether to use audio features
            audio_weight: Influence of audio (0-1) in final highlights
            strict_audio: Whether to fail on audio processing errors
            max_retries: Maximum retries for transient failures
        """
        self.transcriber = transcriber or VideoTranscriber()
        self.summarizer = summarizer or Summarizer()
        self.audio_analyzer = audio_analyzer or AudioAnalyzer(noise_reduce=True)
        self.enable_audio_fusion = enable_audio_fusion
        self.audio_weight = min(max(audio_weight, 0), 1)  # Clamp to 0-1 range
        self.text_weight = 1 - self.audio_weight
        self.strict_audio = strict_audio
        self.max_retries = max_retries

    async def process_video(
        self,
        video_path: Path,
        style: str = "default",
        max_highlights: int = 5,
        overwrite: bool = False,
        **transcribe_kwargs
    ) -> Dict[str, Any]:
        try:
            # Setup paths and validate
            component_dir, txt_path, json_path, audio_json_path = get_output_paths(video_path)
            summary_path = component_dir / f"{video_path.stem}_summary.json"
            
            # Ensure paths are absolute
            component_dir = component_dir.resolve()
            validate_output_paths(txt_path, json_path, overwrite)
            
            if not overwrite and summary_path.exists():
                raise FileExistsError(f"Summary exists: {summary_path}")

            # Step 1: Transcribe video (with retries)
            transcription = await self._retry_operation(
                self.transcriber.process_video,
                video_path,
                overwrite=overwrite,
                **transcribe_kwargs
            )
            logger.info(f"Transcription completed: {len(transcription.segments)} segments")

            # Step 2: Process Audio (if enabled)
            audio_analysis = None
            if self.enable_audio_fusion and transcription.audio_path:
                try:
                    audio_analysis = await self._analyze_audio(
                        audio_path=transcription.audio_path,
                        segments=transcription.segments,
                        component_dir=component_dir
                    )
                    logger.info(f"Audio analysis completed: {len(audio_analysis)} segments")
                except Exception as e:
                    if self.strict_audio:
                        raise PipelineError("Audio processing failed") from e
                    logger.warning(f"Audio analysis failed, proceeding without it: {str(e)}")

            # Step 3: Summarize transcript
            summary = await self._summarize_transcript(
                transcription,  # Pass the full transcription object
                style,
                max_highlights,
                audio_analysis
            )

            # Step 4: Fuse audio features if available
            if audio_analysis:
                summary = self._fuse_audio_features(summary, audio_analysis)

            # Save and return results
            result = self._format_output(
                transcription,
                summary,
                str(component_dir),
                style=style
            )
            await self._atomic_write(summary_path, result)
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed for {video_path}: {str(e)}", exc_info=True)
            raise PipelineError(f"Video processing failed: {str(e)}") from e

    async def _analyze_audio(
        self,
        audio_path: Path,
        segments: List[Dict],
        component_dir: Path
    ) -> Dict[Tuple[float, float], Dict[str, Any]]:
        """
        Analyze audio and return features keyed by (start,end) tuples
        
        Args:
            audio_path: Path to audio file
            segments: Transcript segments to analyze
            component_dir: Directory to save analysis results
            
        Returns:
            Dictionary of {(start,end): audio_features}
        """
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file missing: {audio_path}")
                
            # Batch analyze all segments
            audio_segments = self.audio_analyzer.batch_analyze(audio_path, segments)
            
            if not audio_segments:
                raise PipelineError("No valid audio segments analyzed")
            
            # Save analysis results
            analysis_path = self.audio_analyzer.save_audio_analysis(
                audio_path,
                audio_segments,
                component_dir
            )
            logger.debug(f"Audio analysis saved to: {analysis_path}")
            
            # Create lookup dictionary
            return {
                (round(s.start, 2), round(s.end, 2)): {
                    'confidence': s.confidence,
                    'emotion': s.emotion,
                    'speech_prob': s.speech_prob,
                    'label': s.label,
                    'energy': s.energy,
                    'pitch': s.pitch
                }
                for s in audio_segments
            }
        except Exception as e:
            logger.error("Audio analysis failed", exc_info=True)
            raise PipelineError("Audio processing failed") from e

    def _fuse_audio_features(
        self, 
        summary: SummaryResult, 
        audio_analysis: Dict[Tuple[float, float], Dict[str, Any]]
    ) -> SummaryResult:
        """
        Combine text and audio features to enhance highlights
        
        Args:
            summary: Text summary result
            audio_analysis: Audio features dictionary
            
        Returns:
            Enhanced SummaryResult with combined confidence scores
        """
        for highlight in summary.highlights:
            # Handle both single and multi-segment highlights
            if isinstance(highlight.start, list):
                # Multi-segment highlight - weighted average
                audio_features = []
                for s, e in zip(highlight.start, highlight.end):
                    key = (round(s, 2), round(e, 2))
                    if key in audio_analysis:
                        audio_features.append(audio_analysis[key])
                
                if audio_features:
                    weights = [f['confidence'] for f in audio_features]
                    total_weight = sum(weights) or 1.0  # Avoid division by zero
                    
                    highlight.audio_confidence = sum(
                        f['confidence'] * w for f, w in zip(audio_features, weights)
                    ) / total_weight
                    
                    highlight.audio_metadata = {
                        'emotion': max(
                            set(f['emotion'] for f in audio_features),
                            key=lambda x: sum(
                                f['confidence'] for f in audio_features 
                                if f['emotion'] == x
                            )
                        ),
                        'avg_energy': sum(
                            f['energy'] * w for f, w in zip(audio_features, weights)
                        ) / total_weight,
                        'avg_pitch': sum(
                            f['pitch'] * w for f, w in zip(audio_features, weights)
                        ) / total_weight
                    }
            else:
                # Single segment highlight
                key = (round(highlight.start[0], 2), round(highlight.end[0], 2))
                if key in audio_analysis:
                    audio_data = audio_analysis[key]
                    highlight.audio_confidence = audio_data['confidence']
                    highlight.audio_metadata = {
                        'emotion': audio_data['emotion'],
                        'energy': audio_data['energy'],
                        'pitch': audio_data['pitch']
                    }
            
            # Calculate combined confidence
            if hasattr(highlight, 'audio_confidence'):
                highlight.combined_confidence = (
                    self.text_weight * highlight.confidence + 
                    self.audio_weight * highlight.audio_confidence
                )
            else:
                highlight.combined_confidence = highlight.confidence
        
        # Re-sort highlights by combined confidence
        summary.highlights.sort(key=lambda x: x.combined_confidence, reverse=True)
        return summary

    async def _summarize_transcript(
        self,
        transcription: 'TranscriptionResult',  # Ensure type hint
        style: str,
        max_highlights: int,
        audio_analysis: Optional[Dict] = None
    ) -> 'SummaryResult':
        """Wrapper for summarization with error context"""
        try:
            # Verify we have a proper TranscriptionResult
            if not hasattr(transcription, 'text') or not hasattr(transcription, 'segments'):
                raise ValueError("Invalid transcription object - missing required attributes")
                
            return await self._retry_operation(
                self.summarizer.summarize,
                transcript=transcription.text,
                transcript_segments=transcription.segments,
                style=style,
                max_highlights=max_highlights,
                aligned_segments=list(audio_analysis.values()) if audio_analysis else None
            )
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}", exc_info=True)
            raise PipelineError("Text summarization failed") from e

    def _format_output(
        self,
        transcription: TranscriptionResult,
        summary: SummaryResult,
        component_dir: str,
        style: str = "default"
    ) -> Dict[str, Any]:
        """
        Format final output with all metadata
        
        Args:
            transcription: Raw transcription results
            summary: Processed summary results
            component_dir: Output directory path
            style: Summary style used
            
        Returns:
            Dictionary containing all pipeline outputs
        """
        return {
            "metadata": {
                "pipeline_version": "2.1",
                "processing_time": datetime.now().isoformat(),
                "audio_fusion_enabled": self.enable_audio_fusion,
                "audio_weight": self.audio_weight,
                "text_weight": self.text_weight,
                "style": style
            },
            "transcription": {
                "text": transcription.text,
                "language": transcription.language,
                "duration": transcription.duration,
                "word_count": len(transcription.text.split()),
                "segment_count": len(transcription.segments),
                "audio_path": str(transcription.audio_path) if transcription.audio_path else None,
                "audio_analysis_path": str(Path(component_dir) / f"{Path(transcription.audio_path).stem}_audio_analysis.json") 
                                    if self.enable_audio_fusion and transcription.audio_path else None
            },
            "summary": {
                "style": style,
                "overview": summary.overview,
                "keywords": summary.keywords,
                "highlights": [
                    {
                        "text": h.highlight_text,
                        "original_text": h.original_text,
                        "category": h.category,
                        "timestamps": self._format_timestamps(h),
                        "confidence": round(h.combined_confidence, 3),
                        "text_confidence": round(h.confidence, 3),
                        "audio_confidence": round(getattr(h, 'audio_confidence', 0), 3),
                        "audio_metadata": getattr(h, 'audio_metadata', None)
                    }
                    for h in summary.highlights
                ]
            },
            "paths": {
                "component_dir": component_dir,
                "transcript": str(Path(component_dir) / f"{Path(transcription.audio_path).stem}.txt"),
                "segments": str(Path(component_dir) / f"{Path(transcription.audio_path).stem}_segments.json"),
                "summary": str(Path(component_dir) / f"{Path(transcription.audio_path).stem}_summary.json")
            }
        }

    def _format_timestamps(self, highlight) -> List[Dict]:
        """Convert timestamp lists to formatted dictionaries"""
        if isinstance(highlight.start, list):
            return [
                {"start": round(s, 2), "end": round(e, 2)}
                for s, e in zip(highlight.start, highlight.end)
            ]
        return [{"start": round(highlight.start[0], 2), "end": round(highlight.end[0], 2)}]

    async def _retry_operation(self, func, *args, **kwargs):
        """Retry wrapper with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay}s")
                await asyncio.sleep(delay)

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