def format_timestamp(seconds: float) -> str:
    """
    Converts seconds (e.g., 125.456) to HH:MM:SS.mmm format.
    Example: 125.456 -> '00:02:05.456'
    """
    import math

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
