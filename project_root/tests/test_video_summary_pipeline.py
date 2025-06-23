import asyncio
from pathlib import Path
import json
import pytest
from pipelines.video_summary_pipeline import VideoSummaryPipeline

SAMPLE_VIDEO = Path("data/inputs/sample_video.mp4")
TEST_STYLES = ["default", "journalism", "lecture"]

@pytest.fixture
def pipeline():
    return VideoSummaryPipeline()

@pytest.mark.asyncio
async def test_full_pipeline(pipeline):
    """Test the complete pipeline from video to summary"""
    result = await pipeline.process_video(
        video_path=SAMPLE_VIDEO,
        style="default",
        max_highlights=3,
        overwrite=True
    )
    
    # Verify basic structure
    assert "transcription" in result
    assert "summary" in result
    assert "paths" in result
    
    # Check transcription results
    transcription = result["transcription"]
    assert isinstance(transcription["text"], str)
    assert len(transcription["text"]) > 0
    assert transcription["duration"] > 0
    assert transcription["segment_count"] > 0
    
    # Check summary results
    summary = result["summary"]
    assert len(summary["highlights"]) == 3
    assert isinstance(summary["overview"], str)
    assert len(summary["keywords"]) > 0
    
    # Verify files were created
    paths = result["paths"]
    assert Path(paths["transcript"]).exists()
    assert Path(paths["segments"]).exists()
    assert Path(paths["summary"]).exists()
    
    # Verify summary file content
    with open(paths["summary"]) as f:
        saved_summary = json.load(f)
    assert saved_summary["highlight_count"] == 3

@pytest.mark.asyncio
@pytest.mark.parametrize("style", TEST_STYLES)
async def test_different_styles(pipeline, style):
    """Test pipeline with different summarization styles"""
    result = await pipeline.process_video(
        video_path=SAMPLE_VIDEO,
        style=style,
        max_highlights=2,
        overwrite=True
    )
    
    assert len(result["summary"]["highlights"]) == 2
    assert result["summary"]["model_metadata"]["provider"] in ["openai", "anthropic", "together"]

@pytest.mark.asyncio
async def test_overwrite_protection(pipeline):
    """Test that overwrite=False prevents file overwrites"""
    # First run should succeed
    await pipeline.process_video(
        video_path=SAMPLE_VIDEO,
        overwrite=False
    )
    
    # Second run should fail
    with pytest.raises(FileExistsError):
        await pipeline.process_video(
            video_path=SAMPLE_VIDEO,
            overwrite=False
        )

@pytest.mark.asyncio
async def test_invalid_video_path(pipeline):
    """Test handling of non-existent video file"""
    with pytest.raises(FileNotFoundError):
        await pipeline.process_video(
            video_path=Path("nonexistent.mp4")
        )

@pytest.mark.asyncio
async def test_output_structure(pipeline):
    """Verify the output directory structure"""
    result = await pipeline.process_video(
        video_path=SAMPLE_VIDEO,
        overwrite=True
    )
    
    paths = result["paths"]
    component_dir = Path(paths["component_dir"])
    
    # Verify directory naming pattern
    assert component_dir.name.startswith("sample_video_")
    assert component_dir.parent.name == "components"
    
    # Verify all expected files exist
    assert (component_dir / "sample_video.txt").exists()
    assert (component_dir / "sample_video_segments.json").exists()
    assert (component_dir / "sample_video_summary.json").exists()