import json
import logging
import re
import asyncio
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from summarization.utils.llm.router import (
    route_with_fallback,
    Style,
    LLMRequest,
    LLMResponse
)
from summarization.utils.llm.exceptions import LLMError

logger = logging.getLogger(__name__)

# ------------------ Data Models ------------------

@dataclass
class Highlight:
    original_text: str
    highlight_text: str
    start: List[float]
    end: List[float]
    category: str
    confidence: float = 0.0
    reasoning: Optional[str] = None
    audio_confidence: Optional[float] = None
    combined_confidence: Optional[float] = None
    audio_metadata: Optional[Dict] = None

@dataclass
class SummaryResult:
    highlights: List[Highlight]
    overview: Optional[str] = None
    keywords: Optional[List[str]] = None
    model_metadata: Optional[Dict] = None

# ------------------ Prompt Templates ------------------

class PromptManager:
    def __init__(self, template_dir: str = "summarization/text/prompts"):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )

    def render(
        self,
        style: str,
        transcript: str,
        transcript_segments: List[Dict],  # Required parameter
        max_highlights: int,
        aligned_segments: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """Render prompt with timestamped transcript"""
        # Format timestamped transcript
        timestamped_transcript = "\n\n".join(
            f"[{seg['start']:.2f}s-{seg['end']:.2f}s]\n{seg['text']}"
            for seg in transcript_segments
        )    
        # Format audio segments
        audio_context = []
        if aligned_segments:
            for seg in aligned_segments:
                if isinstance(seg, dict):
                    audio_context.append({
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'emotion': seg.get('emotion', 'neutral'),
                        'energy': seg.get('energy', 0),
                        'confidence': seg.get('confidence', 0),
                        'pitch': seg.get('pitch', 0)
                    })
                elif hasattr(seg, '__dict__'):
                    audio_context.append({
                        'start': seg.start,
                        'end': seg.end,
                        'emotion': seg.emotion,
                        'energy': seg.energy,
                        'confidence': seg.confidence,
                        'pitch': seg.pitch
                    })

        if not transcript.strip():
           raise ValueError("Empty transcript provided")

        prompt = self.env.get_template(f"{style}.jinja2").render(
            full_transcript=transcript,
            timestamped_transcript=timestamped_transcript,
            max_highlights=max_highlights,
            audio_segments=audio_context,
            **kwargs
        )
        
        logger.debug(f"Prompt contains:\n"
                    f"- {len(timestamped_transcript.splitlines())} transcript lines\n"
                    f"- {len(audio_context)} audio segments")
        return prompt

    def _format_timestamped_transcript(self, segments: List[Dict]) -> str:
        """Convert segments to timestamped text blocks"""
        return "\n\n".join(
            f"[{seg['start']:.2f}s - {seg['end']:.2f}s]\n{seg['text']}"
            for seg in segments
        )

# ------------------ Core Summarizer ------------------

class Summarizer:
    def __init__(self, prompt_dir: Optional[str] = None):
        self.prompts = PromptManager(prompt_dir or str(Path(__file__).parent / "prompts"))

    async def summarize(
        self,
        transcript: str,
        transcript_segments: List[Dict],
        style: Union[Style, str] = Style.DEFAULT,
        max_highlights: int = 5,
        aligned_segments: Optional[List[Dict]] = None,
        **kwargs
    ) -> SummaryResult:
        """Generate summary with timestamped context"""
        try:
            if isinstance(style, str):
                style = Style.from_string(style)

            logger.info(
                f"Starting summarization with:\n"
                f"- {len(transcript_segments)} transcript segments\n"
                f"- {len(aligned_segments or [])} audio segments"
            )

            prompt = self.prompts.render(
                style=style.value,
                transcript=transcript,
                transcript_segments=transcript_segments,
                max_highlights=max_highlights,
                aligned_segments=aligned_segments,
                **kwargs
            )

            # Debug output
            debug_path = Path("llm_prompt_debug.txt")
            debug_path.write_text(prompt)
            logger.debug(f"Full prompt saved to: {debug_path}")

            response = await self._safe_llm_call(style, prompt)
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Summarization failed: {str(e)}") from e

    async def _safe_llm_call(self, style: Style, prompt: str) -> LLMResponse:
        """Handle both sync and async LLM providers"""
        llm_request = LLMRequest(
            prompt=prompt,
            system_prompt=f"You are a professional {style.value} content summarizer. "
                          "Use the provided timestamps when creating highlights.",
            json_mode=True,
            temperature=self._get_temperature_for_style(style),
            max_tokens=self._get_max_tokens_for_style(style)
        )
        
        if asyncio.iscoroutinefunction(route_with_fallback):
            return await route_with_fallback(style, llm_request)
        
        result = route_with_fallback(style, llm_request)
        if isinstance(result, LLMResponse):
            return result
        if asyncio.isfuture(result) or asyncio.iscoroutine(result):
            return await result
        
        raise RuntimeError("Unexpected response type from LLM router")

    def _parse_response(self, response: LLMResponse) -> SummaryResult:
        """Parse LLM response with timestamp validation"""
        try:
            data = json.loads(self._clean_response_text(response.text))
            
            highlights = []
            for h in data.get('highlights', []):
                try:
                    if not all(k in h for k in ['original_text', 'highlight_text', 'start', 'end']):
                        continue
                        
                    highlights.append(Highlight(
                        original_text=h['original_text'].strip(),
                        highlight_text=h['highlight_text'].strip(),
                        start=self._convert_timestamps(h['start']),
                        end=self._convert_timestamps(h['end']),
                        category=h.get('category', 'FACT').upper(),
                        confidence=float(h.get('confidence', 0)),
                        reasoning=h.get('reasoning'),
                        audio_confidence=float(h.get('audio_confidence', 0)),
                        combined_confidence=float(h.get('combined_confidence', 0)),
                        audio_metadata=h.get('audio_metadata')
                    ))
                except Exception as e:
                    logger.warning(f"Skipping invalid highlight: {str(e)}")
                    continue

            if not highlights:
                raise ValueError("No valid highlights found in response")

            return SummaryResult(
                highlights=highlights,
                overview=data.get('overview'),
                keywords=data.get('keywords', []),
                model_metadata={
                    'model': response.model,
                    'provider': response.provider.value,
                    'tokens': response.tokens_used,
                    'cost': response.cost,
                    'raw_response': response.text[:1000]
                }
            )
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise ValueError(f"Invalid LLM response: {str(e)}") from e

    def _clean_response_text(self, text: str) -> str:
        """Sanitize LLM response"""
        return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text.strip())

    def _convert_timestamps(self, timestamp: Union[str, float, List]) -> List[float]:
        """Convert timestamps from any format to seconds"""
        if isinstance(timestamp, list):
            return [self._timestamp_to_seconds(t) for t in timestamp]
        return [self._timestamp_to_seconds(timestamp)]

    def _timestamp_to_seconds(self, timestamp: Union[str, float]) -> float:
        """Convert HH:MM:SS.SSS or MM:SS.SSS or float to seconds"""
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
            
        if isinstance(timestamp, str):
            parts = timestamp.replace(',', '.').split(':')
            try:
                if len(parts) == 3:  # HH:MM:SS
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:  # MM:SS
                    return float(parts[0]) * 60 + float(parts[1])
                return float(timestamp)  # Plain seconds
            except ValueError:
                logger.warning(f"Invalid timestamp format: {timestamp}")
                return 0.0
        return 0.0

    def _get_temperature_for_style(self, style: Style) -> float:
        return {
            Style.JOURNALISM: 0.3,
            Style.LECTURE: 0.4,
            Style.DEFAULT: 0.7,
            Style.CELEBRATION: 1.0
        }.get(style, 0.7)

    def _get_max_tokens_for_style(self, style: Style) -> int:
        return {
            Style.JOURNALISM: 2048,
            Style.LECTURE: 3072,
            Style.DEFAULT: 1024,
            Style.SPORTS: 512
        }.get(style, 1024)