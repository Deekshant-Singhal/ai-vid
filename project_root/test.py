import os

print("Current environment variables:")
print(os.environ.get('OPENROUTER_API_KEY', 'NOT FOUND'))

# Alternative way to check
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("After .env load:", os.environ.get('OPENROUTER_API_KEY'))
except ImportError:
    print("python-dotenv not installed")