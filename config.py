from dotenv import load_dotenv
import os

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
