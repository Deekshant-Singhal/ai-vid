import os
from pathlib import Path
from datetime import datetime
from typing import Tuple

def safe_create_directory(dir_path: Path) -> None:
    """Ensures directory exists and is writable with atomic checks"""
    dir_path.mkdir(parents=True, exist_ok=True)
    test_file = dir_path / '.permission_test'
    try:
        test_file.write_text('test')
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Cannot write to directory {dir_path}: {e}")

def generate_video_component_dir(video_path: Path) -> Path:
    """
    Creates: /data/components/[video_stem]_[timestamp]/
    Example: /data/components/interview_20230815_143000/
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    component_dir = Path("data/components") / f"{video_path.stem}_{timestamp}"
    safe_create_directory(component_dir)
    return component_dir

def get_output_paths(video_path: Path) -> Tuple[Path, Path, Path]:
    """
    Returns: (component_dir, txt_path, json_path)
    Example:
    (
        Path("/data/components/interview_20230815_143000"),
        Path("/data/components/interview_20230815_143000/interview.txt"),
        Path("/data/components/interview_20230815_143000/interview_segments.json")
    )
    """
    component_dir = generate_video_component_dir(video_path)
    base_name = video_path.stem
    return (
        component_dir,
        component_dir / f"{base_name}.txt",
        component_dir / f"{base_name}_segments.json"
    )

def validate_output_paths(txt_path: Path, json_path: Path, overwrite: bool = False) -> None:
    """Validates both output paths in one atomic operation"""
    if not overwrite:
        for path in [txt_path, json_path]:
            if path.exists():
                raise FileExistsError(f"File exists and overwrite=False: {path}")
    
    if not os.access(txt_path.parent, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {txt_path.parent}")