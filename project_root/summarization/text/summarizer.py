import json
import logging
import re
import asyncio
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from difflib import SequenceMatcher
import logging
logger = logging.getLogger(__name__)

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
    category: str
    confidence: float = 0.0
    reasoning: Optional[str] = None
    audio_confidence: Optional[float] = None
    combined_confidence: Optional[float] = None
    audio_metadata: Optional[Dict] = None
    semantic_similarity: Optional[float] = None
    bert_score: Optional[float] = None

@dataclass
class SummaryResult:
    highlights: List[Highlight]
    overview: Optional[str] = None
    keywords: Optional[List[str]] = None
    model_metadata: Optional[Dict] = None
    quality_metrics: Optional[Dict] = None

# ------------------ Prompt Templates ------------------

class PromptManager:
    def __init__(self, template_dir: str = "summarization/text/prompts"):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')

    def render(
        self,
        style: str,
        transcript: str,
        transcript_segments: List[Dict],
        max_highlights: int,
        aligned_segments: Optional[List[Dict]] = None,
        highlight_candidates: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """Render prompt with timestamped transcript and optional candidates"""
        # Format timestamped transcript
        timestamped_transcript = self._format_timestamped_transcript(transcript_segments)
        
        # Format audio segments
        audio_context = self._format_audio_segments(aligned_segments)

        if not transcript.strip():
            raise ValueError("Empty transcript provided")

        # Prepare context for refinement passes
        refinement_context = ""
        if highlight_candidates:
            refinement_context = "\nCandidate Highlights:\n" + "\n".join(
                f"- {h['text']} [Score: {h.get('score', 0):.2f}]" 
                for h in highlight_candidates
            )

        template = self.env.get_template(f"{style}.jinja2")
        return template.render(
            full_transcript=transcript,
            timestamped_transcript=timestamped_transcript,
            max_highlights=max_highlights,
            audio_segments=audio_context,
            refinement_context=refinement_context,
            **kwargs
        )

    def _format_timestamped_transcript(self, segments: List[Dict]) -> str:
        """Convert segments to timestamped text blocks with confidence indicators"""
        formatted = []
        for seg in segments:
            confidence = seg.get('confidence', 0)
            confidence_marker = "✓" if confidence > 0.7 else "?" if confidence > 0.5 else "✗"
            formatted.append(
                f"[{seg['start']:.2f}s-{seg['end']:.2f}s] {confidence_marker}\n"
                f"{seg['text']}"
            )
        return "\n\n".join(formatted)

    def _format_audio_segments(self, segments: Optional[List[Dict]]) -> List[Dict]:
        """Format audio segments with emotion and energy indicators"""
        if not segments:
            return []
        
        formatted = []
        for seg in segments:
            if isinstance(seg, dict):
                formatted.append({
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'emotion': seg.get('emotion', 'neutral'),
                    'energy': self._energy_level(seg.get('energy', 0)),
                    'confidence': seg.get('confidence', 0),
                    'pitch': seg.get('pitch', 0)
                })
        return formatted

    def _energy_level(self, energy: float) -> str:
        """Convert energy value to human-readable level"""
        if energy > 0.05: return "high"
        if energy > 0.02: return "medium"
        return "low"

# ------------------ Core Summarizer ------------------

class Summarizer:

    def _map_timestamps(self, highlight: Highlight, segments: List[Dict]) -> Highlight:
        """
        Robust timestamp mapping using nuclear normalization with progressive fallbacks.
        Guarantees matches for identical content regardless of formatting differences.
        """
        import re
        from difflib import SequenceMatcher

        def nuclear_normalize(text: str) -> str:
            """Strip all formatting differences (whitespace/punctuation/case)"""
            return re.sub(r'[^\w]', '', text).lower()  # \w = [a-zA-Z0-9_]

        # Phase 0: Input validation
        if not segments:
            logger.warning("Empty segments list provided")
            highlight.start = [0.0]
            highlight.end = [0.0]
            return highlight

        target = nuclear_normalize(highlight.original_text)
        target_len = len(target)

        # Phase 1: Nuclear exact match (handles 95%+ cases)
        for seg in segments:
            if nuclear_normalize(seg['text']) == target:
                highlight.start = [seg['start']]
                highlight.end = [seg['end']]
                return highlight

        # Phase 2: Nuclear substring match (handles minor LLM edits)
        for seg in segments:
            seg_text = nuclear_normalize(seg['text'])
            if target in seg_text or seg_text in target:
                highlight.start = [seg['start']]
                highlight.end = [seg['end']]
                return highlight

        # Phase 3: Length-aware fuzzy matching
        best_ratio = 0
        best_segment = None
        min_threshold = 0.75 if target_len > 15 else 0.85  # Stricter for short texts

        for seg in segments:
            seg_text = nuclear_normalize(seg['text'])
            ratio = SequenceMatcher(None, target, seg_text).ratio()
            
            # Dynamic threshold based on match quality and segment length
            if ratio > best_ratio and ratio > min_threshold:
                best_ratio = ratio
                best_segment = seg
                if ratio > 0.95:  # Early exit for near-perfect matches
                    break

        if best_segment:
            highlight.start = [best_segment['start']]
            highlight.end = [best_segment['end']]
            logger.debug(f"Fuzzy match succeeded (ratio={best_ratio:.2f}): {highlight.original_text[:50]}...")
            return highlight

        # Phase 4: SBERT semantic fallback (if available)
        if hasattr(self, 'sbert_model'):
            try:
                highlight = self._map_with_sbert(highlight, segments)
                if highlight.start[0] != 0.0:
                    return highlight
            except Exception as e:
                logger.warning(f"SBERT fallback failed: {str(e)}")

        # Phase 5: Final fallback with detailed diagnostics
        sample_segments = [seg['text'] for seg in segments[:3]]
        logger.error(
            f"Timestamp mapping completely failed for:\n"
            f"Original: {highlight.original_text[:200]}...\n"
            f"Normalized: {target[:200]}...\n"
            f"Top 3 segments:\n1. {sample_segments[0][:100]}...\n"
            f"2. {sample_segments[1][:100]}...\n3. {sample_segments[2][:100]}..."
        )
        highlight.start = [0.0]
        highlight.end = [0.0]
        return highlight

    def _map_with_sbert(self, highlight: Highlight, segments: List[Dict]) -> Highlight:
        """Semantic fallback using SBERT embeddings"""
        original_embed = self.sbert_model.encode(highlight.original_text)
        
        best_score = 0
        best_segment = None
        
        for seg in segments:
            seg_embed = self.sbert_model.encode(seg['text'])
            similarity = np.dot(original_embed, seg_embed) / (
                np.linalg.norm(original_embed) * np.linalg.norm(seg_embed))
            
            if similarity > best_score:
                best_score = similarity
                best_segment = seg
                if similarity > 0.9:  # Early exit for strong matches
                    break
        
        if best_score > 0.75:
            highlight.start = [best_segment['start']]
            highlight.end = [best_segment['end']]
            logger.info(f"SBERT fallback matched (score={best_score:.2f})")
        
        return highlight


    def __init__(self, prompt_dir: Optional[str] = None):
        self.prompts = PromptManager(prompt_dir or str(Path(__file__).parent / "prompts"))
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
        self.min_confidence = 0.65  # Minimum confidence threshold
        self.quality_metrics = {
            'min_confidence': self.min_confidence,
            'semantic_weight': 0.6,
            'bert_weight': 0.4
        }

    async def summarize(
        self,
        transcript: str,
        transcript_segments: List[Dict],
        style: Union[Style, str] = Style.DEFAULT,
        max_highlights: int = 5,
        aligned_segments: Optional[List[Dict]] = None,
        two_pass: bool = True,
        **kwargs
    ) -> SummaryResult:
        """Enhanced summarization with two-pass processing"""
        try:
            if isinstance(style, str):
                style = Style.from_string(style)

            logger.info(
                f"Starting summarization with {len(transcript_segments)} segments "
                f"and {len(aligned_segments or [])} audio segments"
            )

            # First pass - broad candidate generation
            first_pass = await self._first_pass_summary(
                transcript,
                transcript_segments,
                style,
                max_highlights * 2,  # Get more candidates
                aligned_segments
            )

            # Second pass - refinement
            if two_pass:
                result = await self._refine_summary(
                    first_pass,
                    transcript,
                    transcript_segments,
                    style,
                    max_highlights,
                    aligned_segments
                )
            else:
                result = first_pass

            # Calculate quality metrics
            result.quality_metrics = self._calculate_quality_metrics(result)

            return result
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Summarization failed: {str(e)}") from e

    async def _first_pass_summary(
        self,
        transcript: str,
        segments: List[Dict],
        style: Style,
        max_highlights: int,
        audio_segments: Optional[List[Dict]]
    ) -> SummaryResult:
        prompt = self.prompts.render(
            style=style.value,
            transcript=transcript,
            transcript_segments=segments,
            max_highlights=max_highlights,
            aligned_segments=audio_segments
        )
        
        response = await self._safe_llm_call(style, prompt)
        return self._parse_response(response, transcript,segments, style)  # Fixed

    async def _safe_llm_call(self, style: Style, prompt: str) -> LLMResponse:
        llm_request = LLMRequest(
            prompt=prompt,
            system_prompt=self._get_system_prompt(style),
            json_mode=True,
            temperature=self._get_temperature_for_style(style),
            max_tokens=self._get_max_tokens_for_style(style)
        )
        
        try:
            # Handle both sync and async versions
            if asyncio.iscoroutinefunction(route_with_fallback):
                response = await route_with_fallback(style, llm_request)
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: route_with_fallback(style, llm_request)
                )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise LLMError(f"LLM processing failed: {str(e)}") from e
    async def _refine_summary(
        self,
        first_pass: SummaryResult,
        transcript: str,
        segments: List[Dict],
        style: Style,
        max_highlights: int,
        audio_segments: Optional[List[Dict]]
    ) -> SummaryResult:
        """Refine highlights based on confidence scores"""
        # Prepare candidate highlights with scores
        candidates = []
        for h in first_pass.highlights:
            candidates.append({
                'text': h.highlight_text,
                'original': h.original_text,
                'score': h.combined_confidence or h.confidence,
                'timestamps': list(zip(h.start, h.end))
            })

        # Generate refinement prompt
        prompt = self.prompts.render(
            style=style.value,
            transcript=transcript,
            transcript_segments=segments,
            max_highlights=max_highlights,
            aligned_segments=audio_segments,
            highlight_candidates=candidates
        )

        response = await self._safe_llm_call(style, prompt)
        return self._parse_response(response, transcript,segments,style)

    def _calculate_quality_metrics(self, result: SummaryResult) -> Dict:
        """Calculate summary quality metrics"""
        if not result.highlights:
            return {'error': 'No highlights generated'}
        
        confidences = [h.combined_confidence or h.confidence for h in result.highlights]
        return {
            'average_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'coverage_score': self._calculate_coverage_score(result),
            'diversity_score': self._calculate_diversity_score(result)
        }

    def _calculate_coverage_score(self, result: SummaryResult) -> float:
        """Estimate how well highlights cover the transcript"""
        # Implementation depends on your specific requirements
        return 0.8  # Placeholder

    def _calculate_diversity_score(self, result: SummaryResult) -> float:
        """Calculate semantic diversity between highlights"""
        if len(result.highlights) < 2:
            return 1.0
            
        embeddings = self.sbert_model.encode(
            [h.highlight_text for h in result.highlights]
        )
        similarities = np.triu(
            np.dot(embeddings, embeddings.T) / (
                np.linalg.norm(embeddings, axis=1)[:, None] * 
                np.linalg.norm(embeddings, axis=1)[None, :]
            ),
            k=1
        )
        return float(1 - np.mean(similarities[similarities > 0]))

    def _parse_response(self, response: LLMResponse, transcript: str, segments: List[Dict], style: Style) -> SummaryResult:
        """Parse LLM response and map timestamps to original text segments"""
        try:
            # 1. Parse and clean response
            data = json.loads(self._clean_response_text(response.text))
            
            highlights = []
            for h in data.get('highlights', []):
                try:
                    # 2. Validate required fields
                    if not all(k in h for k in ['original_text', 'highlight_text']):
                        logger.warning(f"Skipping highlight missing required fields: {h}")
                        continue
                    
                    # 3. Create base highlight (without timestamps)
                    highlight = Highlight(
                        original_text=h['original_text'].strip(),
                        highlight_text=h['highlight_text'].strip(),
                        category=h.get('category', 'FACT').upper(),
                        confidence=float(h.get('confidence', 0)),
                        reasoning=h.get('reasoning'),
                        audio_confidence=float(h.get('audio_confidence', 0)),
                        combined_confidence=float(h.get('combined_confidence', 0)),
                        audio_metadata=h.get('audio_metadata')
                    )
                    
                    # 4. Map timestamps from transcript segments
                    highlight = self._map_timestamps(highlight, segments)
                    
                    # 5. Calculate semantic quality metrics
                    self._calculate_semantic_scores(highlight, transcript)
                    
                    # 6. Apply confidence threshold
                    if (highlight.combined_confidence or highlight.confidence) >= self.min_confidence:
                        highlights.append(highlight)
                        
                except Exception as e:
                    logger.warning(f"Skipping invalid highlight: {str(e)}", exc_info=True)
                    continue

            # 7. Sort and limit highlights
            sorted_highlights = sorted(
                highlights,
                key=lambda x: x.combined_confidence or x.confidence,
                reverse=True
            )[:self._get_max_highlights_for_style(style)]

            return SummaryResult(
                highlights=sorted_highlights,
                overview=data.get('overview'),
                keywords=data.get('keywords', []),
                model_metadata={
                    'model': response.model,
                    'provider': response.provider.value,
                    'tokens': response.tokens_used,
                    'cost': response.cost
                }
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {response.text[:200]}...")
            raise ValueError(f"LLM returned invalid JSON: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}", exc_info=True)
            raise ValueError(f"Response parsing failed: {str(e)}") from e


    def _calculate_semantic_scores(self, highlight: Highlight, transcript: str):
        """Calculate BERT-based semantic scores"""
        # Semantic similarity
        highlight_embed = self.sbert_model.encode(highlight.highlight_text)
        original_embed = self.sbert_model.encode(highlight.original_text)
        highlight.semantic_similarity = float(np.dot(highlight_embed, original_embed) / (
            np.linalg.norm(highlight_embed) * np.linalg.norm(original_embed)))
        
        # BERTScore
        _, _, f1 = bert_score([highlight.highlight_text], [highlight.original_text], lang='en')
        highlight.bert_score = float(f1.mean().item())
        
        # Update confidence if not set
        if highlight.confidence == 0:
            highlight.confidence = (
                self.quality_metrics['semantic_weight'] * highlight.semantic_similarity +
                self.quality_metrics['bert_weight'] * highlight.bert_score
            )

   
    def _get_system_prompt(self, style: Style) -> str:
        """Generate style-specific system prompt"""
        base = (
            f"You are a professional {style.value} content summarizer. "
            "Use the provided timestamps when creating highlights. "
            "Focus on factual accuracy and importance."
        )
        
        if style == Style.JOURNALISM:
            return base + " Prioritize who, what, when, where, why."
        elif style == Style.LECTURE:
            return base + " Emphasize key concepts and definitions."
        return base

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
    def _get_max_highlights_for_style(self, style: Style) -> int:
        """Dynamic highlight count based on style"""
        return {
            Style.JOURNALISM: 5,
            Style.LECTURE: 7,
            Style.DEFAULT: 5,
            Style.SPORTS: 3
        }.get(style, 5)
    
