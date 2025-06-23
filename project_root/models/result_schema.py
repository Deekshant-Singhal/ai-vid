from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class TranscriptionResult:
    text: str
    segments: List[Dict[str, Any]]
    language: Optional[str] = None
    duration: Optional[float] = None