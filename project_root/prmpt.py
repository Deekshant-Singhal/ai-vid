from summarization.text.summarizer import PromptManager

prompt = PromptManager("summarization/text/prompts").render(
    style="journalism",
    transcript="The speaker warned about rising temperatures. A scientist confirmed the risk to polar bears.",
    max_highlights=3,
    aligned_segments=[
        {
            "start": 5.2,
            "end": 10.3,
            "emotion": "urgent",
            "confidence": 0.88,
            "energy": 0.75,
            "pitch": 230.0
        },
        {
            "start": 15.0,
            "end": 20.0,
            "emotion": "neutral",
            "confidence": 0.62,
            "energy": 0.45,
            "pitch": 190.0
        }
    ]
)

print("-------- RENDERED PROMPT --------")
print(prompt)
