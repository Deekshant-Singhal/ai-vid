#!/usr/bin/env python3
import asyncio
from pathlib import Path
import json
from datetime import datetime
from pipelines.video_summary_pipeline import VideoSummaryPipeline

async def main():
    print("Starting journalism-style video summarization test...")
    
    # Setup paths
    output_dir = Path("debug_outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize pipeline
    pipeline = VideoSummaryPipeline()
    sample_video = Path("data/inputs/sample_video.mp4")

    if not sample_video.exists():
        print(f"❌ Error: Sample video not found at {sample_video}")
        return

    print(f"📹 Processing video: {sample_video}")

    try:

        # === Step 1: Transcription ===
        print("\n=== Step 1: Transcribing Video ===")
        transcription = await pipeline.transcriber.process_video(
            video_path=sample_video,
            overwrite=True
        )
        
        # Verify we got a proper TranscriptionResult
        if not hasattr(transcription, 'text') or not hasattr(transcription, 'segments'):
            raise ValueError("Transcription did not return valid TranscriptionResult object")
            
        print(f"Transcription contains {len(transcription.segments)} segments")
        
        # Verify transcription
        if not transcription.text.strip():
            raise ValueError("Empty transcription returned")
        if not transcription.segments:
            raise ValueError("No timed segments in transcription")

        # Save raw transcription
        trans_path = output_dir / f"transcription_{timestamp}.json"
        with open(trans_path, 'w') as f:
            json.dump({
                'text': transcription.text,
                'segments': transcription.segments,
                'audio_path': str(transcription.audio_path) if transcription.audio_path else None
            }, f, indent=2)
        print(f"💾 Saved raw transcription to: {trans_path}")

        # === Step 2: Analyze Audio ===
        print("\n=== Step 2: Analyzing Audio ===")
        if not transcription.audio_path:
            raise ValueError("No audio path in transcription results")
            
        audio_analysis = await pipeline._analyze_audio(
            audio_path=Path(transcription.audio_path),
            segments=transcription.segments,
            component_dir=output_dir
        )
        
        # Convert tuple keys to strings for JSON serialization
        serializable_audio_analysis = {
            f"{start}_{end}": values
            for (start, end), values in audio_analysis.items()
        }
        
        # Save audio analysis
        audio_path = output_dir / f"audio_analysis_{timestamp}.json"
        with open(audio_path, 'w') as f:
            json.dump(serializable_audio_analysis, f, indent=2)
        print(f"💾 Saved audio analysis to: {audio_path}")

        # === Step 3: Generate Prompt ===
        print("\n=== Step 3: Generating Prompt ===")
        prompt = pipeline.summarizer.prompts.render(
            style="journalism",
            transcript=transcription.text,
            transcript_segments=transcription.segments,  # Critical addition
            max_highlights=3,
            aligned_segments=list(audio_analysis.values())
        )
        
        # Save raw prompt
        prompt_path = output_dir / f"llm_prompt_{timestamp}.txt"
        with open(prompt_path, 'w') as f:
            f.write("=== SYSTEM PROMPT ===\n")
            f.write(f"You are a professional journalism content summarizer\n\n")
            f.write("=== USER PROMPT ===\n")
            f.write(prompt)
        
        print(f"💾 Saved raw prompt to: {prompt_path}")
        print("\n=== Prompt Preview ===")
        print(prompt[:500] + "...")  # Show first 500 chars

        # === Step 4: Run Full Pipeline ===
        print("\n=== Step 4: Running Full Pipeline ===")
        result = await pipeline.process_video(
            video_path=sample_video,
            style="journalism",
            max_highlights=3,
            overwrite=True
        )

        # === Step 5: Save Results ===
        print("\n=== Final Results ===")
        result_path = output_dir / f"results_{timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Saved full results to: {result_path}")

        # Print highlights
        print("\n🎯 Key Highlights:")
        for i, h in enumerate(result['summary']['highlights'], 1):
            print(f"\n{i}. [{h['timestamps'][0]['start']:.2f}s-{h['timestamps'][0]['end']:.2f}s]")
            print(f"   Category: {h['category']}")
            print(f"   Text: {h['text']}")
            if 'audio_confidence' in h:
                print(f"   Audio Confidence: {h['audio_confidence']:.2f}")

        print("\n✅ Test completed successfully!")
        print(f"📂 All debug files saved to: {output_dir}")

    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())