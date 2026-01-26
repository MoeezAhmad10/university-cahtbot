import os
from dotenv import load_dotenv

load_dotenv()

print("OPENAI_API_KEY from environment:", "✓ FOUND" if os.getenv("OPENAI_API_KEY") else "✗ NOT FOUND")
print("First 10 chars:", os.getenv("OPENAI_API_KEY", "")[:10] if os.getenv("OPENAI_API_KEY") else "N/A")