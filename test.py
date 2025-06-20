from transcribe import get_transcript_from_video

video_path = "sample_video.mp4"
text, segments = get_transcript_from_video(video_path)

print("\nFull Transcript:\n", text)
print("\n--- Segments (start & end times) ---")
for segment in segments:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
