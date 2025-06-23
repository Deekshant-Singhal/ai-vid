import os
import json
import requests
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from summarization.utils.llm.exceptions import LLMTimeoutError, LLMAPIError

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # second

@dataclass
class OpenRouterParams:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    stop_sequences: Optional[List[str]] = None
    json_mode: bool = True  # OpenRouter supports native JSON mode
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

def call_openrouter(
    prompt: str,
    model: str,
    system_message: Optional[str] = None,  # ✅ Corrected
    params: OpenRouterParams = OpenRouterParams(),
    expected_structure: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    OpenRouter API client with advanced features:
    
    - Native JSON response mode
    - Model routing (automatically selects provider)
    - Cost tracking headers
    - Structured output validation
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",  # Required for some models
        "X-Title": "Video Highlight Generator"      # Appears in dashboard
    }

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
    }

    if params.stop_sequences:
        payload["stop"] = params.stop_sequences
    if params.json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise LLMAPIError(data["error"].get("message", "OpenRouter error"))

            if not data.get("choices"):
                raise LLMAPIError("No choices in response")

            content = data["choices"][0]["message"]["content"]
            
            try:
                parsed = json.loads(content)
                if expected_structure and not all(key in parsed for key in expected_structure):
                    raise LLMAPIError(f"Response missing required keys: {expected_structure}")
                return {
                    **parsed,
                    "_metadata": {
                        "model": data.get("model"),
                        "usage": data.get("usage"),
                        "cost": response.headers.get("x-openrouter-cost")
                    }
                }
            except json.JSONDecodeError as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise LLMAPIError(f"Invalid JSON response: {str(e)}\nContent: {content}")

        except requests.exceptions.Timeout:
            raise LLMTimeoutError(timeout_duration=TIMEOUT)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise LLMAPIError(f"OpenRouter API request failed: {str(e)}") from e

    raise LLMAPIError(f"Max retries exceeded. Last error: {str(last_error)}")
