#!/usr/bin/env python3
"""
FIXED startup script for MAI-DxO Web UI
Uses server_fixed.py which delays mai_dx import to avoid Windows hang
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("\n" + "="*60)
    print("   MAI-DxO Web UI - Fixed Windows Version")
    print("="*60)
    
    # Check for API keys
    has_keys = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GEMINI_API_KEY"),  
        os.getenv("ANTHROPIC_API_KEY")
    ])
    
    if not has_keys:
        print("\n⚠️  No API keys detected in environment")
        print("   You can set them in the web interface")
    else:
        print("\n✅ API keys detected")
    
    print("\n🚀 Starting server...")
    print("   URL: http://127.0.0.1:8000")
    print("   Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Open browser after delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open("http://127.0.0.1:8000")
            print("🌐 Browser opened")
        except:
            pass
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start the fixed server
    try:
        # Windows event loop setup
        if sys.platform == "win32":
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            print("✅ Windows event loop configured")
        
        import uvicorn
        print("✅ Uvicorn imported")
        
        # Import the FIXED server that delays mai_dx import
        from webui.server_fixed import app
        print("✅ Server app loaded (mai_dx will load on first use)")
        
        # Load saved API keys if they exist
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
        
        print("🎯 Server starting...\n")
        
        # Run with asyncio loop explicitly
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=8000, 
            log_level="info",
            loop="asyncio"  # Force asyncio instead of uvloop
        )
        
    except ImportError as e:
        print(f"\n❌ Missing package: {e}")
        print("\n📦 Install requirements:")
        print("   pip install fastapi uvicorn")
        input("\nPress Enter to exit...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
