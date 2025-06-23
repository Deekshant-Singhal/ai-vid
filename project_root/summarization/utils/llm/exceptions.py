class LLMError(Exception):
    """Base exception for all LLM-related errors"""
    def __init__(self, message: str, original_exception: Exception = None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(message)

    def __str__(self):
        if self.original_exception:
            return f"{self.message} (Original: {type(self.original_exception).__name__}: {str(self.original_exception)})"
        return self.message

class LLMTimeoutError(LLMError):
    """Raised when an LLM API request times out"""
    def __init__(self, timeout_duration: float, original_exception: Exception = None):
        message = f"LLM API request timed out after {timeout_duration} seconds"
        super().__init__(message, original_exception)
        self.timeout_duration = timeout_duration

class LLMAPIError(LLMError):
    """Raised for API-related failures (non-200 responses, etc.)"""
    def __init__(
        self,
        message: str,
        status_code: int = None,
        api_response: dict = None,
        original_exception: Exception = None
    ):
        self.status_code = status_code
        self.api_response = api_response
        
        full_message = message
        if status_code:
            full_message += f" (Status: {status_code})"
        if api_response and 'error' in api_response:
            full_message += f" | API Error: {api_response['error']}"
            
        super().__init__(full_message, original_exception)

class LLMContentError(LLMError):
    """Raised for content-related issues (malformed JSON, invalid schema, etc.)"""
    def __init__(
        self,
        message: str,
        invalid_content: str = None,
        validation_errors: list = None,
        original_exception: Exception = None
    ):
        self.invalid_content = invalid_content
        self.validation_errors = validation_errors or []
        
        full_message = message
        if validation_errors:
            full_message += f"\nValidation errors:\n- " + "\n- ".join(validation_errors)
            
        super().__init__(full_message, original_exception)

class LLMRateLimitError(LLMAPIError):
    """Raised when rate limits are exceeded"""
    def __init__(
        self,
        limit: str = None,
        reset_time: float = None,
        original_exception: Exception = None
    ):
        message = "LLM API rate limit exceeded"
        if limit:
            message += f" (Limit: {limit})"
        if reset_time:
            message += f" | Resets in {reset_time:.0f} seconds"
            
        super().__init__(message, original_exception=original_exception)
        self.limit = limit
        self.reset_time = reset_time