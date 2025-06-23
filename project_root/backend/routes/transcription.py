from fastapi import APIRouter, UploadFile, File
# from transcribe.video_transcriber import VideoTranscriber
# from pathlib import Path
# import shutil
# import asyncio

router = APIRouter()

# @router.post("/")
# async def transcribe_video(file: UploadFile = File(...)):
#     temp_path = Path(f"data/inputs/{file.filename}")
#     with open(temp_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     transcriber = VideoTranscriber()
#     result = await transcriber.process_video(temp_path)

#     return {
#         "text": result.text,
#         "segments": result.segments,
#         "duration": result.duration,
#         "language": result.language,
#     }
