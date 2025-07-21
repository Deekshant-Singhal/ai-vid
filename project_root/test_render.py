import asyncio
import json
from summarization.text.summarizer import Summarizer
from summarization.utils.llm.router import LLMResponse, Provider, Style

async def test_parse_response():
    # 1. Initialize your actual summarizer
    summarizer = Summarizer()
    
    # 2. Create test data matching your production format
    test_segments = [
        {"text": "Plastic pollution is a global crisis.", "start": 10.2, "end": 12.5, "confidence": 0.9},
        {"text": "Only 9% of plastic gets recycled.", "start": 30.1, "end": 32.8, "confidence": 0.8}
    ]
    
    # 3. Mock API response (must match your LLM's actual JSON output)
    mock_response = LLMResponse(
        text=json.dumps({
            "highlights": [
                {
                    "original_text": "Plastic pollution is a global crisis.",
                    "highlight_text": "Global plastic crisis",
                    "category": "FACT",
                    "confidence": 0.85
                },
                {
                    "original_text": "Only 9% of plastic gets recycled.",
                    "highlight_text": "Low recycling rates",
                    "category": "STAT"
                }
            ],
            "overview": "Plastic pollution overview",
            "keywords": ["plastic", "recycling"]
        }),
        model="mistralai/mixtral-8x7b-instruct",
        provider=Provider.OPENROUTER,
        tokens_used=150
    )

    # 4. Test the actual function
    result = summarizer._parse_response(
        response=mock_response,
        transcript="Full transcript text...",
        segments=test_segments,
        style=Style.JOURNALISM
    )

    # 5. Verify results
    assert len(result.highlights) == 2
    assert result.highlights[0].start == [10.2]  # Should match first segment
    assert result.highlights[1].category == "STAT"
    assert "plastic" in result.keywords
    
    print("✅ Test passed! Parsed highlights:")
    for h in result.highlights:
        print(f"- {h.highlight_text} ({h.start[0]}s)")

if __name__ == "__main__":
    asyncio.run(test_parse_response())