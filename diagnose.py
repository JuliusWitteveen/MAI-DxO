"""Quick diagnostic to find where the server is failing"""
import sys

print("Step 1: Python version check")
print(f"   Python: {sys.version}")

print("\nStep 2: Import asyncio")
try:
    import asyncio
    print("   ✅ asyncio imported")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\nStep 3: Set Windows event loop policy")
try:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("   ✅ Windows event loop policy set")
    else:
        print("   ⚠️  Not Windows")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\nStep 4: Import FastAPI")
try:
    from fastapi import FastAPI
    print("   ✅ FastAPI imported")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\nStep 5: Import uvicorn") 
try:
    import uvicorn
    print("   ✅ uvicorn imported")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\nStep 6: Import mai_dx")
try:
    from mai_dx.main import MaiDxOrchestrator
    print("   ✅ MAI-DxO backend imported")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    print("   This might be causing the hang!")

print("\nStep 7: Create minimal FastAPI app")
try:
    test_app = FastAPI()
    
    @test_app.get("/test")
    def test_endpoint():
        return {"status": "ok"}
    
    print("   ✅ Test app created")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n✅ All diagnostic steps completed!")
print("\nIf the script hangs after this, the issue is in the server startup.")
input("\nPress Enter to exit...")
