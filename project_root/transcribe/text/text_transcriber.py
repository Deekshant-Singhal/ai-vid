import whisper # type: ignore
from typing import Optional, Tuple, List, Dict, Any
from config import WHISPER_MODEL


class TextTranscriber:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize Whisper model with optional device specification.
        
        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.model = whisper.load_model(WHISPER_MODEL, device=device)
        self._verify_model_loaded()

    def _verify_model_loaded(self):
        """Verify the model loaded correctly."""
        if not hasattr(self.model, 'transcribe'):
            raise RuntimeError("Whisper model failed to load properly")

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        temperature: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6),
        beam_size: Optional[int] = 5,
        best_of: Optional[int] = 5,
        patience: Optional[float] = None,
        length_penalty: Optional[float] = None,
        suppress_tokens: Optional[List[int]] = [-1],
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Enhanced audio transcription with Whisper.

        Args:
            audio_path: Path to audio file
            language: ISO language code (e.g. 'en')
            temperature: Tuple of temperatures for sampling
            beam_size: Number of beams in beam search
            best_of: Number of candidates when sampling
            patience: Beam search patience factor
            length_penalty: Alpha parameter for length penalty
            suppress_tokens: Tokens to suppress (-1 suppresses most special chars)
            initial_prompt: Optional initial text prompt
            condition_on_previous_text: Whether to use previous text as prompt
            word_timestamps: Whether to include word-level timestamps
            prepend_punctuations: Punctuations to prepend to next word
            append_punctuations: Punctuations to append to previous word

        Returns:
            Tuple of (full_text, segments) where segments contains:
            - text: segment text
            - start: start time in seconds
            - end: end time in seconds
            - [words]: word-level timestamps if word_timestamps=True
        """
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                verbose=False,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                suppress_tokens=suppress_tokens,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=word_timestamps,
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations
            )
            
            # Verify the result structure
            if not isinstance(result, dict) or 'text' not in result:
                raise RuntimeError("Unexpected transcription result format")
                
            return result["text"], result.get("segments", [])
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}") from e

    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Transcribe multiple audio files efficiently.
        
        Args:
            audio_paths: List of audio file paths
            language: Optional language code
            **kwargs: Additional transcribe_audio arguments
            
        Returns:
            List of (text, segments) tuples
        """
        return [self.transcribe_audio(path, language, **kwargs) 
                for path in audio_paths]

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return sorted(list(whisper.tokenizer.LANGUAGES.keys()))

    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language code is supported."""
        return language_code in whisper.tokenizer.LANGUAGES