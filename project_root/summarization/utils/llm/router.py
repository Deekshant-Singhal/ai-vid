import os
import time
import random
import json
import logging
from typing import Literal, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial

# Import clients and exceptions
from summarization.utils.llm.clients.openrouter import call_openrouter, OpenRouterParams
from summarization.utils.llm.exceptions import (
    LLMTimeoutError,
    LLMAPIError,
    LLMRateLimitError,
    LLMContentError
)

logger = logging.getLogger(__name__)

# ------------------ Enums & Types ------------------

class Provider(str, Enum):
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"  # Kept as available through OpenRouter

class Style(str, Enum):
    DEFAULT = "default"
    JOURNALISM = "journalism"
    LECTURE = "lecture"
    SPORTS = "sports"
    FASHION = "fashion"
    INTERVIEW = "interview"
    CELEBRATION = "celebration"

    @classmethod
    def from_string(cls, style_str: str) -> 'Style':
        """Convert string to Style enum, case-insensitive"""
        try:
            return cls(style_str.lower())
        except ValueError:
            valid_styles = [s.value for s in cls]
            raise ValueError(f"Invalid style '{style_str}'. Valid styles: {valid_styles}")

@dataclass
class ModelConfig:
    provider: Provider
    model: str
    max_tokens: int = 2048
    preferred: bool = False  # Marks primary provider for a style

@dataclass
class LLMRequest:
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None  # Overrides ModelConfig if set
    top_p: float = 0.9
    json_mode: bool = False  # For structured output
    stop_sequences: Optional[List[str]] = None

@dataclass
class LLMResponse:
    text: str
    model: str
    provider: Provider
    tokens_used: int
    cost: Optional[float] = None
    latency: Optional[float] = None  # Seconds
    is_fallback: bool = False

# ------------------ Configuration ------------------

# Updated STYLE_MODEL_MAP with free models
STYLE_MODEL_MAP: Dict[Style, List[ModelConfig]] = {
    Style.DEFAULT: [
        ModelConfig(Provider.OPENROUTER, "mistralai/mixtral-8x7b-instruct", preferred=True),  # Free tier available
        ModelConfig(Provider.OPENROUTER, "huggingfaceh4/zephyr-7b-beta"),
        ModelConfig(Provider.OPENROUTER, "gryphe/mythomax-l2-13b")
    ],
    Style.JOURNALISM: [
        ModelConfig(Provider.OPENROUTER, "mistralai/mixtral-8x7b-instruct", preferred=True),
        ModelConfig(Provider.OPENROUTER, "nousresearch/nous-hermes-2-mixtral-8x7b-dpo")
    ],
    Style.LECTURE: [
        ModelConfig(Provider.OPENROUTER, "anthropic/claude-instant-v1", preferred=True),  # Free tier available
        ModelConfig(Provider.OPENROUTER, "openchat/openchat-7b")
    ],
    Style.SPORTS: [
        ModelConfig(Provider.OPENROUTER, "mistralai/mistral-7b-instruct", preferred=True),  # Free tier available
        ModelConfig(Provider.OPENROUTER, "pygmalionai/mythalion-13b")
    ],
    Style.FASHION: [
        ModelConfig(Provider.OPENROUTER, "gryphe/mythomax-l2-13b", preferred=True),
        ModelConfig(Provider.OPENROUTER, "undi95/remm-slerp-l2-13b")
    ],
    Style.INTERVIEW: [
        ModelConfig(Provider.OPENROUTER, "nousresearch/nous-hermes-2-mixtral-8x7b-dpo", preferred=True),
        ModelConfig(Provider.OPENROUTER, "openchat/openchat-7b")
    ],
    Style.CELEBRATION: [
        ModelConfig(Provider.OPENROUTER, "mistralai/mixtral-8x7b-instruct", preferred=True),
        ModelConfig(Provider.OPENROUTER, "pygmalionai/mythalion-13b")
    ]
}

# ------------------ Circuit Breaker ------------------

class CircuitBreaker:
    def __init__(self, fail_threshold: int = 3, cooldown: int = 60):
        self.fail_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.fail_threshold = fail_threshold
        self.cooldown = timedelta(seconds=cooldown)

    def record_failure(self):
        self.fail_count += 1
        self.last_failure_time = datetime.now()
        logger.warning(f"CircuitBreaker: Failure recorded (count={self.fail_count})")

    def is_open(self) -> bool:
        if self.fail_count < self.fail_threshold:
            return False
        if self.last_failure_time and datetime.now() - self.last_failure_time < self.cooldown:
            return True
        # Reset if cooldown period has passed
        self.fail_count = 0
        return False

breakers = defaultdict(CircuitBreaker)

# ------------------ Provider Handlers ------------------

def handle_openrouter(config: ModelConfig, request: LLMRequest) -> LLMResponse:
    start_time = time.monotonic()
    params = OpenRouterParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or config.max_tokens,
        stop_sequences=request.stop_sequences,
        json_mode=request.json_mode
    )
    response = call_openrouter(
        prompt=request.prompt,
        model=config.model,
        system_message=request.system_prompt,
        params=params
    )

    # Extract metadata from OpenRouter response
    usage = response.get("_metadata", {}).get("usage", {})
    cost = response.get("_metadata", {}).get("cost")
    model_name = response.get("_metadata", {}).get("model")

    return LLMResponse(
        text=json.dumps({k: v for k, v in response.items() if k != "_metadata"}),  # Convert back to string
        model=model_name or config.model,
        provider=config.provider,
        tokens_used=usage.get("total_tokens", 0),
        cost=float(cost) if cost else None,
        latency=time.monotonic() - start_time
    )

PROVIDER_HANDLERS = {
    Provider.OPENROUTER: handle_openrouter,
}

# ------------------ Router Core ------------------

def route_with_fallback(
    style: Union[Style, str],
    request: LLMRequest,
    retries: int = 3,
    allow_fallback: bool = True
) -> LLMResponse:
    """
    Smart LLM routing with:
    - Model fallback
    - Circuit breaking
    - Cost tracking
    - Latency monitoring
    
    Args:
        style: Content style (as Style enum or string)
        request: LLM request parameters
        retries: Max attempts (including fallbacks)
        allow_fallback: Whether to try fallback models
    
    Returns:
        LLMResponse with metadata
    
    Raises:
        RuntimeError: When all attempts fail
        ValueError: For invalid configurations
    """
    # Convert string style to enum if needed
    if isinstance(style, str):
        style = Style.from_string(style)

    if style not in STYLE_MODEL_MAP:
        raise ValueError(f"Unknown style: {style}. Available: {list(STYLE_MODEL_MAP.keys())}")

    # Get preferred and fallback configs
    configs = STYLE_MODEL_MAP[style]
    preferred = next((c for c in configs if c.preferred), configs[0])
    fallbacks = [c for c in configs if not c.preferred] if allow_fallback else []
    
    # Prepare attempt order (preferred first, then shuffled fallbacks)
    attempts = [preferred] + random.sample(fallbacks, len(fallbacks))
    attempts = attempts[:retries]
    
    logger.info(f"Routing request for style '{style.value}'. Attempt order: {[a.model for a in attempts]}")

    last_error = None
    for i, config in enumerate(attempts):
        if breakers[config.provider].is_open():
            logger.warning(f"Skipping {config.provider} due to circuit breaker")
            continue

        try:
            handler = PROVIDER_HANDLERS.get(config.provider)
            if not handler:
                raise ValueError(f"No handler for provider {config.provider}")

            response = handler(config, request)
            response.is_fallback = i > 0  # Mark if this was a fallback attempt
            
            logger.info(
                f"Success with {config.provider}/{config.model} "
                f"(tokens: {response.tokens_used}, cost: {response.cost or 'N/A'}, "
                f"latency: {response.latency:.2f}s)"
            )
            return response

        except LLMRateLimitError as e:
            breakers[config.provider].record_failure()
            wait_time = e.reset_time or 5
            logger.warning(f"Rate limited on {config.provider}. Waiting {wait_time}s...")
            time.sleep(wait_time)
        except (LLMTimeoutError, LLMAPIError) as e:
            breakers[config.provider].record_failure()
            last_error = e
            logger.warning(f"Attempt {i+1} failed with {config.provider}: {str(e)}")
        except Exception as e:
            breakers[config.provider].record_failure()
            last_error = e
            logger.error(f"Unexpected error with {config.provider}: {str(e)}", exc_info=True)

    raise RuntimeError(
        f"All {retries} attempts failed for style '{style.value}'. Last error: {str(last_error)}"
    ) from last_error