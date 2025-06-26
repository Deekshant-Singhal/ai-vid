import librosa
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import noisereduce as nr
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    """Dataclass representing analyzed audio segment features"""
    start: float
    end: float
    energy: float               # Root mean square energy (loudness)
    zcr: float                  # Zero-crossing rate (noisiness)
    pitch: float                # Spectral centroid (Hz)
    confidence: float           # Overall quality confidence (0-1)
    speech_prob: float          # Probability of speech content (0-1)
    emotion: Optional[str] = None  # Estimated emotion label
    label: Optional[str] = None    # Audio type classification

class AudioAnalyzer:
    """Analyzes audio files to extract meaningful features for summarization"""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        frame_length: int = 2048,
        hop_length: int = 512,
        noise_reduce: bool = True,
        min_confidence: float = 0.4
    ):
        """
        Initialize audio analyzer with processing parameters
        
        Args:
            sample_rate: Target sample rate for analysis
            frame_length: Number of samples per analysis frame
            hop_length: Number of samples between frames
            noise_reduce: Whether to apply noise reduction
            min_confidence: Minimum confidence threshold for valid segments
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.noise_reduce = noise_reduce
        self.min_confidence = min_confidence

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file with optional noise reduction
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            if self.noise_reduce:
                y = nr.reduce_noise(y=y, sr=sr)
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {str(e)}")
            raise

    def analyze_segment(self, audio_path: Path, start: float, end: float) -> AudioSegment:
        """
        Analyze a specific time segment of audio
        
        Args:
            audio_path: Path to audio file
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            AudioSegment with extracted features
        """
        try:
            y, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                offset=start,
                duration=end-start
            )
            if self.noise_reduce:
                y = nr.reduce_noise(y=y, sr=sr)
                
            features = self._extract_enhanced_features(y, sr)
            return AudioSegment(
                start=start,
                end=end,
                **features
            )
        except Exception as e:
            logger.error(f"Failed to analyze segment {start}-{end}s: {str(e)}")
            return AudioSegment(
                start=start,
                end=end,
                energy=0,
                zcr=0,
                pitch=0,
                confidence=0,
                speech_prob=0,
                emotion="unknown",
                label="error"
            )

    def batch_analyze(self, audio_path: Path, segments: List[Dict]) -> List[AudioSegment]:
        """
        Analyze multiple segments from a transcript
        
        Args:
            audio_path: Path to audio file
            segments: List of transcript segments with 'start'/'end' times
            
        Returns:
            List of analyzed AudioSegments
        """
        results = []
        for seg in segments:
            try:
                results.append(self.analyze_segment(
                    audio_path,
                    float(seg['start']),
                    float(seg['end'])
                ))
            except Exception as e:
                logger.warning(f"Skipping segment {seg['start']}-{seg['end']}: {str(e)}")
                continue
        return [s for s in results if s.confidence >= self.min_confidence]

    def save_audio_analysis(
        self, 
        audio_path: Path, 
        segments: List[AudioSegment], 
        component_dir: Path
    ) -> Path:
        """
        Save analysis results to JSON in the component directory
        
        Args:
            audio_path: Source audio file path
            segments: List of analyzed audio segments
            component_dir: Directory to save analysis results
            
        Returns:
            Path to saved analysis file
        """
        try:
            # Ensure directory exists
            component_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output path
            json_path = component_dir / f"{audio_path.stem}_audio_analysis.json"
            
            # Prepare analysis data
            analysis_data = {
                "metadata": {
                    "source_audio": str(audio_path.resolve()),
                    "sample_rate": self.sample_rate,
                    "analysis_time": datetime.now().isoformat(),
                    "analyzer_version": "1.0"
                },
                "segments": [{
                    "start": seg.start,
                    "end": seg.end,
                    "energy": seg.energy,
                    "pitch": seg.pitch,
                    "confidence": seg.confidence,
                    "speech_prob": seg.speech_prob,
                    "emotion": seg.emotion,
                    "label": seg.label,
                    "zcr": seg.zcr
                } for seg in segments]
            }
            
            # Atomic write operation
            tmp_path = json_path.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            tmp_path.replace(json_path)
            
            logger.info(f"Saved audio analysis to {json_path}")
            return json_path
            
        except Exception as e:
            logger.error(f"Failed to save audio analysis: {str(e)}")
            raise RuntimeError(f"Could not save audio analysis: {str(e)}")

    def _extract_enhanced_features(self, y: np.ndarray, sr: int) -> Dict:
        """Core feature extraction from audio samples"""
        # Temporal features
        rms = librosa.feature.rms(
            y=y, 
            frame_length=self.frame_length,
            hop_length=self.hop_length
        ).mean()
        
        zcr = librosa.feature.zero_crossing_rate(
            y, 
            frame_length=self.frame_length,
            hop_length=self.hop_length
        ).mean()
        
        # Spectral features
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        ).mean()
        
        # Higher-level features
        speech_prob = self._estimate_speech_probability(y, sr)
        emotion = self._estimate_emotion(y, sr)
        label = self._classify_audio_type(y, sr)
        
        # Combined confidence score
        confidence = self._calculate_confidence(
            rms=rms,
            zcr=zcr,
            pitch=centroid,
            speech_prob=speech_prob
        )
        
        return {
            'energy': float(rms),
            'zcr': float(zcr),
            'pitch': float(centroid),
            'confidence': float(confidence),
            'speech_prob': float(speech_prob),
            'emotion': emotion,
            'label': label
        }

    def _calculate_confidence(self, rms: float, zcr: float, pitch: float, speech_prob: float) -> float:
        """Calculate overall segment confidence score"""
        return float(
            0.4 * rms +                  # Energy importance
            0.2 * (1 - zcr) +            # Less noise is better
            0.2 * (pitch / 5000) +       # Normalized pitch
            0.2 * speech_prob            # Speech likelihood
        )

    def _estimate_speech_probability(self, y: np.ndarray, sr: int) -> float:
        """Estimate probability of speech content"""
        flatness = librosa.feature.spectral_flatness(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        
        # Simple heuristic (replace with ML model in production)
        if flatness < 0.5 and zcr < 0.2:
            return 0.9  # Clear speech
        elif flatness < 0.7 and zcr < 0.3:
            return 0.6  # Probable speech
        return 0.1      # Unlikely speech

    def _estimate_emotion(self, y: np.ndarray, sr: int) -> str:
        """Classify emotional tone (simplified - enhance with ML)"""
        energy = librosa.feature.rms(y=y).mean()
        pitch = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        
        if energy > 0.1 and pitch > 500:
            return "excited"
        elif energy > 0.1 and pitch < 300:
            return "serious"
        elif energy > 0.05:
            return "neutral"
        return "calm"

    def _classify_audio_type(self, y: np.ndarray, sr: int) -> str:
        """Classify segment type (speech/music/etc)"""
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        
        if self._estimate_speech_probability(y, sr) > 0.7:
            return "speech"
        elif bandwidth > 2000:
            return "music"
        elif bandwidth > 1000:
            return "noise"
        return "silence"

    def detect_key_audio_events(self, audio_path: Path) -> List[AudioSegment]:
        """Auto-detect important segments (for future use)"""
        y, sr = self.load_audio(audio_path)
        duration = len(y) / sr
        
        # Placeholder implementation - enhance with proper event detection
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=self.hop_length)
        
        # Simple peak detection
        peaks = np.where(rms > np.percentile(rms, 75))[0]
        segments = []
        
        for i in peaks:
            start = max(0, times[i] - 0.5)
            end = min(duration, times[i] + 0.5)
            segments.append(self.analyze_segment(audio_path, start, end))
            
        return sorted(segments, key=lambda x: x.confidence, reverse=True)[:20]