# test_video_summary_pipeline.py

import asyncio
from pathlib import Path
import json
from datetime import datetime
import logging
from pipelines.video_summary_pipeline import VideoSummaryPipeline, PipelineError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PromptCaptureSummarizer:
    """Wrapper to capture prompts sent to the LLM"""
    def __init__(self, original_summarizer):
        self.original = original_summarizer
        self.last_prompt = None
        self.last_system_prompt = None
    
    async def summarize(self, *args, **kwargs):
        """Capture prompt before sending to actual summarizer"""
        self.last_prompt = kwargs.get('transcript', args[0] if args else None)
        self.last_system_prompt = kwargs.get('system_prompt', "No system prompt captured")
        
        # Save the prompt immediately
        prompt_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": self.last_prompt,
            "system_prompt": self.last_system_prompt,
            "style": kwargs.get('style', 'default'),
            "max_highlights": kwargs.get('max_highlights', 5),
            "has_audio_data": bool(kwargs.get('aligned_segments'))
        }
        
        prompt_file = Path("data/outputs/llm_prompts.json")
        prompt_file.parent.mkdir(exist_ok=True)
        with open(prompt_file, 'w') as f:
            json.dump(prompt_data, f, indent=2)
        
        logger.info(f"Saved LLM prompt to {prompt_file}")
        
        # Call original summarizer
        return await self.original.summarize(*args, **kwargs)

async def test_video_summary_pipeline():
    """End-to-end test of VideoSummaryPipeline with sample video"""
    print("\n" + "="*50)
    print("Testing VideoSummaryPipeline with sample_video.mp4")
    print("="*50 + "\n")
    
    # Initialize pipeline with audio analysis enabled
    pipeline = VideoSummaryPipeline(enable_audio_fusion=True)
    
    # Wrap the summarizer to capture prompts
    pipeline.summarizer = PromptCaptureSummarizer(pipeline.summarizer)
    
    # Test file paths
    test_video = Path("data/inputs/sample_video.mp4")
    output_dir = Path("data/outputs")
    output_dir.mkdir(exist_ok=True)
    
    if not test_video.exists():
        raise FileNotFoundError(f"Test video not found at {test_video}")

    try:
        logger.info(f"Starting processing of {test_video.name}")
        start_time = datetime.now()
        
        # Run the pipeline
        result = await pipeline.process_video(
            video_path=test_video,
            style="journalism",
            max_highlights=5,
            overwrite=True
        )
        
        # Calculate processing time
        proc_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing completed in {proc_time:.2f} seconds")
        
        # Verify basic output structure
        required_keys = {"metadata", "transcription", "summary", "paths"}
        assert all(k in result for k in required_keys), "Missing result keys"
        
        # Save full results
        result_file = output_dir / "test_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Full results saved to {result_file}")

        # Print summary
        print("\nTEST SUMMARY:")
        print(f"Video Duration: {result['transcription']['duration']:.2f}s")
        print(f"Transcript Length: {result['transcription']['word_count']} words")
        print(f"Highlights Generated: {len(result['summary']['highlights'])}")
        
        # Print first highlight with audio features
        if result['summary']['highlights']:
            hl = result['summary']['highlights'][0]
            print("\nSAMPLE HIGHLIGHT:")
            print(f"Text: {hl['text']}")
            print(f"Time: {hl['timestamps'][0]['start']:.1f}-{hl['timestamps'][0]['end']:.1f}s")
            print(f"Confidence: {hl['confidence']:.2f}")
            if hl.get('audio_metadata'):
                print(f"Emotion: {hl['audio_metadata']['emotion']}")
                print(f"Speech Probability: {hl['audio_metadata']['speech_probability']:.2f}")
        
        return result

    except PipelineError as pe:
        logger.error(f"Pipeline error: {pe}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    print("VideoSummaryPipeline Test Runner")
    print("=" * 40)
    try:
        asyncio.run(test_video_summary_pipeline())
        print("\nTEST PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        exit(1)