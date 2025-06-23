import os
import json
import requests
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from summarization.utils.llm.exceptions import LLMTimeoutError, LLMAPIError
from functools import wraps
import time

TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # second

@dataclass
class TogetherParams:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    stop_sequences: Optional[list] = None
    json_mode: bool = False  # New: Enable JSON response formatting

def retry_on_json_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                continue
        raise LLMAPIError(f"Failed after {MAX_RETRIES} attempts. Last error: {str(last_error)}") from last_error
    return wrapper

def validate_json_structure(response: Dict[str, Any], expected_keys: list) -> bool:
    """Check if response contains required top-level keys"""
    return all(key in response for key in expected_keys)

@retry_on_json_error
def call_together(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    params: TogetherParams = TogetherParams(),
    expected_structure: Optional[list] = None  # e.g., ["highlights", "summary"]
) -> Dict[str, Any]:
    """
    Enhanced Together.ai API caller with JSON validation.
    
    Features:
    - Automatic JSON parsing with retries
    - Optional structure validation
    - JSON mode support
    - Detailed error handling
    """
    if not TOGETHER_API_KEY:
        raise EnvironmentError("TOGETHER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    full_prompt = (system_prompt + "\n\n" + prompt) if system_prompt else prompt

    payload = {
        "model": model,
        "prompt": full_prompt,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "stop": params.stop_sequences,
    }

    # Enable JSON mode if supported by the model
    if params.json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(
            TOGETHER_API_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise LLMAPIError("No choices returned in API response")

        generated_text = data["choices"][0]["text"].strip()

        try:
            parsed = json.loads(generated_text)
            if expected_structure and not validate_json_structure(parsed, expected_structure):
                raise LLMAPIError(f"Response missing required keys: {expected_structure}")
            return parsed
        except json.JSONDecodeError as e:
            raise LLMAPIError(f"Failed to parse JSON response: {str(e)}\nResponse: {generated_text}")

    except requests.exceptions.Timeout:
        raise LLMTimeoutError(f"API request timed out after {TIMEOUT} seconds")
    except requests.exceptions.RequestException as e:
        raise LLMAPIError(f"API request failed: {str(e)}")