#!/usr/bin/env python3
"""
Quick start script for MAI-DxO Web UI
Starts the server immediately without dependency checks
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("\n" + "="*50)
    print("   MAI-DxO Diagnostic System - Web UI")
    print("="*50)
    
    # Check if API keys are configured
    has_keys = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY")
    ])
    
    if not has_keys:
        print("\n⚠️  WARNING: No API keys detected!")
        print("   Set one of these environment variables:")
        print("   - OPENAI_API_KEY")
        print("   - GEMINI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("\n   Or configure them in the web interface.")
    
    print("\n🚀 Starting server...")
    print("   URL: http://127.0.0.1:8000")
    print("   Press Ctrl+C to stop\n")
    print("="*50 + "\n")
    
    # Try to open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except:
            pass
    
    # Start browser opener in background
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start the server
    try:
        # Fix Windows compatibility issue with uvloop
        if sys.platform == "win32":
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        import uvicorn
        from webui.server import app
        
        # Load any saved API keys
        keys_file = Path(__file__).parent / "webui" / "api_keys.json"
        if keys_file.exists():
            import json
            try:
                with open(keys_file, 'r') as f:
                    keys = json.load(f)
                    for key_name, key_value in keys.items():
                        if key_value:
                            os.environ[key_name] = key_value
                print("✅ Loaded saved API keys\n")
            except:
                pass
        
        # Disable uvloop on Windows
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", loop="asyncio")
        
    except ImportError as e:
        print(f"\n❌ ERROR: Missing required package: {e}")
        print("\n📦 Please install requirements:")
        print("   pip install fastapi uvicorn")
        print("\n   Or run: pip install -r requirements.txt")
        input("\nPress Enter to exit...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
