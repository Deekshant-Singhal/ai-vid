# summarization/audio/audio_analyzer.py

import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import noisereduce as nr
import soundfile as sf

@dataclass
class AudioSegment:
    start: float
    end: float
    energy: float
    zcr: float
    pitch: float
    confidence: float
    speech_prob: float  # Probability of speech presence
    emotion: Optional[str] = None  # 'neutral', 'happy', 'angry', etc.
    label: Optional[str] = None  # 'speech', 'music', 'noise', etc.

class AudioAnalyzer:
    def __init__(
        self,
        sample_rate: int = 22050,
        frame_length: int = 2048,
        hop_length: int = 512,
        noise_reduce: bool = True
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.noise_reduce = noise_reduce

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio with optional noise reduction"""
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        if self.noise_reduce:
            y = nr.reduce_noise(y=y, sr=sr)
        return y, sr

    def analyze_segment(self, audio_path: Path, start: float, end: float) -> AudioSegment:
        """Analyze audio segment with enhanced features"""
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

    def _extract_enhanced_features(self, y: np.ndarray, sr: int) -> Dict:
        """Comprehensive feature extraction for speech analysis"""
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
        
        # Speech probability (simplified)
        speech_prob = self._estimate_speech_probability(y, sr)
        
        # Emotion estimation (placeholder - integrate your ML model here)
        emotion = self._estimate_emotion(y, sr)
        
        # Audio type classification
        label = self._classify_audio_type(y, sr)
        
        # Enhanced confidence calculation
        confidence = (
            0.4 * rms +
            0.2 * (1 - zcr) +
            0.2 * (centroid / 5000) +
            0.2 * speech_prob
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

    def _estimate_speech_probability(self, y: np.ndarray, sr: int) -> float:
        """Simple speech/non-speech classifier"""
        # Extract features for speech detection
        spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        
        # Heuristic speech probability (replace with ML model for production)
        if spectral_flatness < 0.5 and zcr < 0.2:
            return 0.9  # Likely speech
        elif spectral_flatness < 0.7 and zcr < 0.3:
            return 0.6  # Possibly speech
        return 0.1  # Unlikely speech

    def _estimate_emotion(self, y: np.ndarray, sr: int) -> str:
        """Placeholder for emotion detection"""
        # TODO: Integrate your emotion detection ML model
        # This is a simplified placeholder
        pitch = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        energy = librosa.feature.rms(y=y).mean()
        
        if energy > 0.1 and pitch > 500:
            return "excited"
        elif energy > 0.05:
            return "neutral"
        return "calm"

    def _classify_audio_type(self, y: np.ndarray, sr: int) -> str:
        """Classify audio segment type"""
        # TODO: Enhance with proper audio classification
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        
        if self._estimate_speech_probability(y, sr) > 0.7:
            return "speech"
        elif spectral_bandwidth > 2000:
            return "music"
        return "other"

    def batch_analyze(self, audio_path: Path, segments: List[Dict]) -> List[AudioSegment]:
        """Batch analyze with progress tracking"""
        results = []
        for i, seg in enumerate(segments):
            try:
                results.append(self.analyze_segment(audio_path, seg['start'], seg['end']))
                print(f"Processed segment {i+1}/{len(segments)}", end='\r')
            except Exception as e:
                print(f"\nError processing segment {seg}: {str(e)}")
                continue
        return results

    def detect_key_audio_events(self, audio_path: Path) -> List[AudioSegment]:
        """Automatically detect important audio segments"""
        y, sr = self.load_audio(audio_path)
        
        # Detect speech regions
        speech_intervals = self._detect_speech_intervals(y, sr)
        
        # Detect emotional peaks
        emotional_segments = self._detect_emotional_peaks(y, sr)
        
        return sorted(
            speech_intervals + emotional_segments,
            key=lambda x: x.confidence,
            reverse=True
        )

    def _detect_speech_intervals(self, y: np.ndarray, sr: int) -> List[AudioSegment]:
        """VAD-like speech detection"""
        # TODO: Implement proper VAD or use library
        pass

    def _detect_emotional_peaks(self, y: np.ndarray, sr: int) -> List[AudioSegment]:
        """Detect emotionally salient segments"""
        # TODO: Implement emotion-based segmentation
        pass