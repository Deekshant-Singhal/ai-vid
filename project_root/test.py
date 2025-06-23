import os
import json
from datetime import datetime
from pathlib import Path
from summarization.text.summarizer import Summarizer

# Configuration
OUTPUT_DIR = Path("output/response")
SAMPLE_TRANSCRIPTS = [
    {
        "title": "AI Podcast Discussion",
        "text": """Artificial intelligence is transforming industries. In healthcare, AI can analyze medical images faster than humans. 
        Machine learning, a subset of AI, enables systems to learn from data. Deep learning uses neural networks with multiple layers.
        The future of AI includes more personalized medicine and autonomous vehicles."""
    },
    {
        "title": "Climate Change Report",
        "text": """Global temperatures have risen 1.1°C since pre-industrial times. The Paris Agreement aims to limit warming to 1.5°C.
        Key solutions include renewable energy adoption and carbon capture. Extreme weather events are becoming more frequent.
        Individual actions like reducing meat consumption can help."""
    }
]

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")

def run_summarization_tests():
    """Run summarization on sample texts and save results"""
    summarizer = Summarizer()
    
    for idx, transcript in enumerate(SAMPLE_TRANSCRIPTS, 1):
        print(f"\n🔍 Processing transcript {idx}: {transcript['title']}")
        
        try:
            # Run summarization
            result = summarizer.summarize(
                transcript=transcript["text"],
                style="journalism",
                max_highlights=3
            )
            
            # Prepare output data
            output_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "transcript_source": transcript["title"],
                    "word_count": len(transcript["text"].split()),
                    "model_metadata": result.model_metadata
                },
                "summary": {
                    "overview": result.overview,
                    "keywords": result.keywords
                },
                "highlights": [
                    {
                        "text": h.highlight_text,
                        "original_text": h.original_text,
                        "start": h.start,
                        "end": h.end,
                        "start_fmt": f"{h.start:.1f}s" if isinstance(h.start, (int, float)) else None,
                        "end_fmt": f"{h.end:.1f}s" if isinstance(h.end, (int, float)) else None,
                        "category": h.category
                    } for h in result.highlights
                ]
            }
            
            # Save outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"summary_{idx}_{timestamp}"
            
            # JSON output
            json_path = OUTPUT_DIR / f"{base_filename}.json"
            with open(json_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"💾 Saved JSON output: {json_path}")
            
            # Human-readable output
            txt_path = OUTPUT_DIR / f"{base_filename}.txt"
            with open(txt_path, "w") as f:
                f.write(f"=== {transcript['title']} ===\n\n")
                f.write(f"Overview:\n{result.overview}\n\n")
                f.write("Highlights:\n")
                for i, h in enumerate(result.highlights, 1):
                    start = f"{h.start:.1f}s" if isinstance(h.start, (int, float)) else "N/A"
                    end = f"{h.end:.1f}s" if isinstance(h.end, (int, float)) else "N/A"
                    f.write(f"{i}. {h.highlight_text} ({start}-{end})\n")
                f.write(f"\nKeywords: {', '.join(result.keywords or [])}\n")
                f.write(f"\nModel: {result.model_metadata.get('model', 'N/A')}")
                f.write(f"\nTokens used: {result.model_metadata.get('tokens', 'N/A')}")
                cost = result.model_metadata.get("cost")
                f.write(f"\nCost: ${cost:.6f}" if isinstance(cost, (int, float)) else "\nCost: N/A")
            print(f"📝 Saved readable output: {txt_path}")
            
        except Exception as e:
            print(f"❌ Failed to process transcript {idx}: {str(e)}")
            continue

if __name__ == "__main__":
    ensure_output_dir()
    run_summarization_tests()
    print("\n✅ Test completed. Check output/response/ for results")
