import os

WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'data/outputs')