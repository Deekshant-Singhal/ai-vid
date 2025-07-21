import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
import transformers
transformers.logging.set_verbosity_error()
from transcribe.video_transcriber import VideoTranscriber, TranscriptionResult
from summarization.text.summarizer import Summarizer, SummaryResult
from paths import get_output_paths, validate_output_paths
from summarization.audio.audio_analyzer import AudioAnalyzer
from summarization.utils.llm.router import Style

logger = logging.getLogger(__name__)

class VideoSummaryPipeline:
    """Enhanced video summarization pipeline with audio-text fusion and quality metrics"""
    
    def __init__(
        self,
        *,
        transcriber: Optional[VideoTranscriber] = None,
        summarizer: Optional[Summarizer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        enable_audio_fusion: bool = True,
        audio_weight: float = 0.3,
        strict_audio: bool = False,
        max_retries: int = 3,
        bert_model: str = 'all-mpnet-base-v2'
    ):
        """
        Initialize pipeline components with enhanced capabilities
        
        Args:
            transcriber: Video transcription component
            summarizer: Text summarization component
            audio_analyzer: Audio feature extraction component
            enable_audio_fusion: Whether to use audio features
            audio_weight: Influence of audio (0-1) in final highlights
            strict_audio: Whether to fail on audio processing errors
            max_retries: Maximum retries for transient failures
            bert_model: SentenceTransformer model for semantic analysis
        """
        self.transcriber = transcriber or VideoTranscriber()
        self.summarizer = summarizer or Summarizer()
        self.audio_analyzer = audio_analyzer or AudioAnalyzer(noise_reduce=True)
        self.enable_audio_fusion = enable_audio_fusion
        self.audio_weight = min(max(audio_weight, 0), 1)  # Clamp to 0-1 range
        self.text_weight = 1 - self.audio_weight
        self.strict_audio = strict_audio
        self.max_retries = max_retries
        self.sbert_model = SentenceTransformer(bert_model)
        
        # Emotion impact factors
        self.EMOTION_WEIGHTS = {
            'excited': 1.3,
            'angry': 1.2, 
            'happy': 1.1,
            'calm': 1.0,
            'sad': 0.9
        }

    async def process_video(
        self,
        video_path: Path,
        style: str = "default",
        max_highlights: int = 5,
        overwrite: bool = False,
        **transcribe_kwargs
    ) -> Dict[str, Any]:
        """Enhanced video processing pipeline with quality tracking"""
        try:
            # Setup paths and validate
            component_dir, txt_path, json_path, audio_json_path = get_output_paths(video_path)
            summary_path = component_dir / f"{video_path.stem}_summary.json"
            
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

            # Step 3: Summarize with two-pass approach
            summary = await self._summarize_transcript(
                transcription,
                style,
                max_highlights,
                audio_analysis
            )

            # Step 4: Enhanced audio-text fusion
            if audio_analysis:
                summary = self._fuse_audio_features(summary, audio_analysis)
                summary = self._apply_temporal_analysis(summary, audio_analysis)

            # Final output formatting
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
        """Enhanced audio analysis with temporal patterns"""
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
            
            # Create enhanced lookup dictionary with temporal features
            return {
                (round(s.start, 2), round(s.end, 2)): {
                    'confidence': s.confidence,
                    'emotion': s.emotion,
                    'speech_prob': s.speech_prob,
                    'label': s.label,
                    'energy': s.energy,
                    'pitch': s.pitch,
                    'zcr': getattr(s, 'zcr', 0)  # Zero-crossing rate if available
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
        """Enhanced audio-text fusion with emotion weighting"""
        if not audio_analysis:
            logger.warning("No audio analysis data available - skipping fusion")
            return summary

        for highlight in summary.highlights:
            # Initialize default audio metadata if missing
            if not hasattr(highlight, 'audio_metadata'):
                highlight.audio_metadata = {
                    'emotion': 'calm',
                    'energy': 0,
                    'pitch': 0,
                    'zcr': 0
                }
                highlight.audio_confidence = 0

            try:
                # Multi-segment highlight handling
                if isinstance(highlight.start, list):
                    audio_features = []
                    for s, e in zip(highlight.start, highlight.end):
                        key = (round(float(s), 2), round(float(e), 2))
                        if key in audio_analysis:
                            audio_features.append(audio_analysis[key])
                    
                    if audio_features:
                        weights = [float(f.get('confidence', 0)) for f in audio_features]
                        total_weight = sum(weights) or 1.0
                        
                        highlight.audio_confidence = sum(
                            float(f.get('confidence', 0)) * w 
                            for f, w in zip(audio_features, weights)
                        ) / total_weight
                        
                        highlight.audio_metadata = {
                            'emotion': max(
                                set(f.get('emotion', 'calm') for f in audio_features),
                                key=lambda x: sum(
                                    f.get('confidence', 0) 
                                    for f in audio_features 
                                    if f.get('emotion', None) == x
                                )
                            ),
                            'avg_energy': sum(
                                float(f.get('energy', 0)) * w 
                                for f, w in zip(audio_features, weights)
                            ) / total_weight,
                            'avg_pitch': sum(
                                float(f.get('pitch', 0)) * w 
                                for f, w in zip(audio_features, weights)
                            ) / total_weight,
                            'avg_zcr': sum(
                                float(f.get('zcr', 0)) * w 
                                for f, w in zip(audio_features, weights)
                            ) / total_weight
                        }

                # Single segment highlight
                else:
                    key = (round(float(highlight.start[0]), 2), 
                          round(float(highlight.end[0]), 2))
                    if key in audio_analysis:
                        audio_data = audio_analysis[key]
                        highlight.audio_confidence = float(audio_data.get('confidence', 0))
                        highlight.audio_metadata = {
                            'emotion': audio_data.get('emotion', 'calm'),
                            'energy': float(audio_data.get('energy', 0)),
                            'pitch': float(audio_data.get('pitch', 0)),
                            'zcr': float(audio_data.get('zcr', 0))
                        }

                # Safe confidence fusion
                base_confidence = (
                    self.text_weight * float(highlight.confidence) + 
                    self.audio_weight * float(getattr(highlight, 'audio_confidence', 0))
                )
                
                emotion = 'calm' if highlight.audio_metadata is None else str(highlight.audio_metadata.get('emotion', 'calm')).lower()
                highlight.combined_confidence = base_confidence * self.EMOTION_WEIGHTS.get(emotion, 1.0)

            except Exception as e:
                logger.error(f"Error processing highlight: {str(e)}", exc_info=True)
                highlight.combined_confidence = highlight.confidence  # Fallback

        # Final sorting
        summary.highlights.sort(key=lambda x: float(getattr(x, 'combined_confidence', x.confidence)), 
                              reverse=True)
        return summary

 
    def _apply_temporal_analysis(
        self, 
        summary: SummaryResult,
        audio_analysis: Dict[Tuple[float, float], Dict[str, Any]]
    ) -> SummaryResult:
        """Apply temporal pattern analysis to boost important sections"""
        # Convert to time series
        times = []
        pitches = []
        energies = []
        for (start, end), features in audio_analysis.items():
            times.append((start + end) / 2)  # Midpoint
            pitches.append(features['pitch'])
            energies.append(features['energy'])
        
        # Calculate z-scores
        pitch_z = (np.array(pitches) - np.mean(pitches)) / np.std(pitches)
        energy_z = (np.array(energies) - np.mean(energies)) / np.std(energies)
        
        # Boost highlights in high-variation regions
        for highlight in summary.highlights:
            if not highlight.audio_metadata:
                continue
                
            # Get midpoint of highlight
            if isinstance(highlight.start, list):
                midpoint = (sum(highlight.start) + sum(highlight.end)) / (2 * len(highlight.start))
            else:
                midpoint = (highlight.start[0] + highlight.end[0]) / 2
            
            # Find nearest analysis point
            idx = np.argmin(np.abs(np.array(times) - midpoint))
            
            # Boost if in high-variation region
            if abs(pitch_z[idx]) > 1.5 or abs(energy_z[idx]) > 1.5:
                highlight.combined_confidence *= 1.2
                highlight.audio_metadata['temporal_boost'] = True
        
        return summary

    async def _summarize_transcript(
        self,
        transcription: TranscriptionResult,
        style: str,
        max_highlights: int,
        audio_analysis: Optional[Dict] = None
    ) -> SummaryResult:
        try:
            if isinstance(style, str):
                style = Style.from_string(style)
                
            return await self.summarizer.summarize(
                transcript=transcription.text,
                transcript_segments=transcription.segments,
                style=style,
                max_highlights=max_highlights,
                aligned_segments=list(audio_analysis.values()) if audio_analysis else None,
                two_pass=True
            )
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise PipelineError(f"Text summarization failed: {str(e)}") from e
    def _format_output(
        self,
        transcription: TranscriptionResult,
        summary: SummaryResult,
        component_dir: str,
        style: str = "default"
    ) -> Dict[str, Any]:
        """Enhanced output formatting with quality metrics"""
        return {
            "metadata": {
                "pipeline_version": "2.2",
                "processing_time": datetime.now().isoformat(),
                "audio_fusion_enabled": self.enable_audio_fusion,
                "audio_weight": self.audio_weight,
                "text_weight": self.text_weight,
                "style": style,
                "quality_metrics": summary.quality_metrics
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
                        "audio_metadata": getattr(h, 'audio_metadata', None),
                        "semantic_similarity": getattr(h, 'semantic_similarity', None),
                        "bert_score": getattr(h, 'bert_score', None)
                    }
                    for h in summary.highlights
                ],
                "quality_metrics": summary.quality_metrics
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