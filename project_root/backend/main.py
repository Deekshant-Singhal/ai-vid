from fastapi import FastAPI
from backend.routes import transcription, summarization

app = FastAPI()

app.include_router(transcription.router, prefix="/api/transcribe")
app.include_router(summarization.router, prefix="/api/summarize")
