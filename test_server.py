"""
Minimal test to check if the server can start on Windows
"""
import sys
import os

# Fix Windows compatibility first
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Add webui compatibility shim
sys.path.insert(0, str(os.path.dirname(__file__)))
from webui.swarms_compat import ensure_windows_compatibility
ensure_windows_compatibility()

print("✅ Windows compatibility configured")

# Try importing the server
try:
    from webui.server import app
    print("✅ Server app imported successfully")
except Exception as e:
    print(f"❌ Failed to import server: {e}")
    sys.exit(1)

# Try starting uvicorn
try:
    import uvicorn
    print("✅ Uvicorn imported")
    print("\n🚀 Starting server at http://127.0.0.1:8000")
    print("   Press Ctrl+C to stop\n")
    
    # Start with explicit asyncio loop
    uvicorn.run(app, host="127.0.0.1", port=8000, loop="asyncio")
    
except Exception as e:
    print(f"❌ Server failed: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
