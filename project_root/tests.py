#!/usr/bin/env python3
import asyncio
from pathlib import Path
from pipelines.video_summary_pipeline import VideoSummaryPipeline
from summarization.text.summarizer import Summarizer  # Access prompt rendering

async def main():
    print("Starting journalism-style video summarization test...")

    # Setup
    pipeline = VideoSummaryPipeline()
    summarizer = pipeline.summarizer
    sample_video = Path("data/inputs/sample_video.mp4")

    if not sample_video.exists():
        print(f"Error: Sample video not found at {sample_video}")
        return

    print(f"Processing video: {sample_video}")

    try:
        # === Step 1: Transcription ===
        print("\n=== Step 1: Transcribing Video ===")
        transcription = await pipeline.transcriber.process_video(
            video_path=sample_video,
            overwrite=True
        )

        # === Step 2: Prompt Generation ===
        print("\n=== Step 2: Generating Prompt ===")
        prompt = summarizer.prompts.render(
            style="journalism",
            transcript=transcription.text,
            max_highlights=3
        )

        component_dir = Path("data/components") / sample_video.stem
        prompt_path = component_dir 
        prompt_path.write_text(prompt)
        print(f"Saved raw prompt to: {prompt_path}")

        # === Step 3: Run Pipeline ===
        print("\n=== Step 3: Running Full Pipeline ===")
        result = await pipeline.process_video(
            video_path=sample_video,
            style="journalism",
            max_highlights=3,
            overwrite=True
        )

        # === Step 4: Show Highlights ===
        print("\n=== Journalism Highlights ===")
        for i, highlight in enumerate(result['summary']['highlights'], 1):
            print(f"\nHighlight {i}:")
            print(f"Text: {highlight['text']}")
            print(f"Timestamps: {highlight['timestamps']}")
            print(f"Category: {highlight['category']}")
            print(f"Confidence: {highlight['confidence']}")

        print("\n=== Saved Files ===")
        print(f"Transcript: {result['paths']['transcript']}")
        print(f"Prompt: {prompt_path}")
        print(f"Summary: {result['paths']['summary']}")
        if 'audio_analysis' in result['paths']:
            print(f"Audio Analysis: {result['paths']['audio_analysis']}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
