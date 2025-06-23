import json
import logging
import re
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
    start: List[float]  # support merged segments
    end: List[float]
    category: str
    confidence: Optional[float] = None

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
        max_highlights: int,
        **kwargs
    ) -> str:
        template = self.env.get_template(f"{style}.jinja2")
        return template.render(
            transcript=transcript,
            max_highlights=max_highlights,
            **kwargs
        )

# ------------------ Core Summarizer ------------------

class Summarizer:
    def __init__(self, prompt_dir: Optional[str] = None):
        self.prompts = PromptManager(prompt_dir or str(Path(__file__).parent / "prompts"))

    def summarize(
        self,
        transcript: str,
        style: Union[Style, str] = Style.DEFAULT,
        max_highlights: int = 5,
        **kwargs
    ) -> SummaryResult:
        try:
            if isinstance(style, str):
                style = Style.from_string(style)

            prompt = self.prompts.render(
                style=style.value,
                transcript=transcript,
                max_highlights=max_highlights,
                **kwargs
            )

            request = LLMRequest(
                prompt=prompt,
                system_prompt=f"You are a professional {style.value} content summarizer",
                json_mode=True,
                temperature=self._get_temperature_for_style(style),
                max_tokens=self._get_max_tokens_for_style(style)
            )

            response = route_with_fallback(style, request)
            return self._parse_response(response)

        except LLMError as e:
            logger.error(f"LLM summarization failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during summarization: {str(e)}", exc_info=True)
            raise RuntimeError(f"Summarization failed: {str(e)}") from e

    def _parse_response(self, response: LLMResponse) -> SummaryResult:
        def safe_json_loads(data: str):
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed for: {data[:200]}...")
                raise

        cleaned_text = response.text.strip()
        cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)

        parsing_attempts = [
            lambda: safe_json_loads(cleaned_text),
            lambda: safe_json_loads(re.search(r'```(?:json)?\n?(.*?)```', cleaned_text, re.DOTALL).group(1).strip()),
            lambda: safe_json_loads(re.search(r'(\[.*\]|\{.*\})', cleaned_text, re.DOTALL).group(0))
        ]

        parsed_data = None
        last_error = None

        for attempt in parsing_attempts:
            try:
                parsed_data = attempt()
                break
            except (json.JSONDecodeError, AttributeError) as e:
                last_error = e
                continue

        if not parsed_data:
            raise ValueError(f"Failed to parse LLM response. Last error: {str(last_error)}\nResponse content: {cleaned_text[:500]}...")

        logger.warning(f"\U0001F50D Cleaned LLM Response:\n{cleaned_text[:1000]}")

        if isinstance(parsed_data, dict):
            highlights_data = parsed_data.get('highlights', [])
            overview = parsed_data.get('overview')
            keywords = parsed_data.get('keywords', [])
        elif isinstance(parsed_data, list):
            highlights_data = parsed_data
            overview = None
            keywords = []
        else:
            raise ValueError(f"Unexpected response format: {type(parsed_data)}")

        highlights = []
        for idx, h in enumerate(highlights_data):
            try:
                if not isinstance(h, dict):
                    logger.warning(f"Highlight {idx} is not a dict: {type(h)}")
                    continue

                required_fields = {'original_text', 'highlight_text', 'start', 'end'}
                if not required_fields.issubset(h.keys()):
                    missing = required_fields - set(h.keys())
                    logger.warning(f"Highlight {idx} missing fields: {missing}")
                    continue

                start_raw = h['start']
                end_raw = h['end']

                start_list = (
                    [self._normalize_timestamp(ts) for ts in start_raw]
                    if isinstance(start_raw, list)
                    else [self._normalize_timestamp(start_raw)]
                )

                end_list = (
                    [self._normalize_timestamp(ts) for ts in end_raw]
                    if isinstance(end_raw, list)
                    else [self._normalize_timestamp(end_raw)]
                )

                highlights.append(Highlight(
                    original_text=str(h['original_text']).strip(),
                    highlight_text=str(h['highlight_text']).strip(),
                    start=start_list,
                    end=end_list,
                    category=str(h.get('category', 'FACT')).upper(),
                    confidence=float(h.get('confidence', 0.0)) if 'confidence' in h else None
                ))
            except Exception as e:
                logger.warning(f"Skipping invalid highlight {idx}: {str(e)}")
                continue

        if not highlights:
            raise ValueError("No valid highlights could be extracted from response")

        return SummaryResult(
            highlights=highlights,
            overview=overview,
            keywords=keywords,
            model_metadata={
                'model': response.model,
                'provider': response.provider.value,
                'tokens': response.tokens_used,
                'cost': response.cost,
                'raw_response': cleaned_text[:1000]
            }
        )

    def _normalize_timestamp(self, timestamp: Union[str, float]) -> float:
        if isinstance(timestamp, (int, float)):
            return float(timestamp)

        try:
            if isinstance(timestamp, str) and ':' in timestamp:
                parts = timestamp.split(':')
                if len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
            return float(timestamp)
        except ValueError:
            logger.warning(f"Invalid timestamp format: {timestamp}")
            return 0.0

    def _get_temperature_for_style(self, style: Style) -> float:
        temps = {
            Style.JOURNALISM: 0.3,
            Style.LECTURE: 0.4,
            Style.DEFAULT: 0.7,
            Style.CELEBRATION: 1.0
        }
        return temps.get(style, 0.7)

    def _get_max_tokens_for_style(self, style: Style) -> int:
        tokens = {
            Style.JOURNALISM: 2048,
            Style.LECTURE: 3072,
            Style.DEFAULT: 1024,
            Style.SPORTS: 512
        }
        return tokens.get(style, 1024)