import os
from dotenv import load_dotenv

load_dotenv()

# Check which APIs are available
apis = {
    "OpenAI": os.getenv("OPENAI_API_KEY"),
    "Gemini": os.getenv("GEMINI_API_KEY"),
    "Anthropic": os.getenv("ANTHROPIC_API_KEY")
}

print("Available APIs:")
for name, key in apis.items():
    if key:
        print(f"✓ {name}: {key[:20]}...")
    else:
        print(f"✗ {name}: Not configured")

# Test import
try:
    from mai_dx import MaiDxOrchestrator
    print("\n✓ MAI-DxO package imported successfully")
except ImportError:
    print("\n✗ MAI-DxO not installed. Run: pip install -r requirements.txt")