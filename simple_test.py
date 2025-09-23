"""Minimal test to find what's causing the hang"""
import sys
print(f"Python {sys.version}")
print("=" * 50)

print("1. Testing basic imports...")
try:
    import asyncio
    print("   ✓ asyncio")
except Exception as e:
    print(f"   ✗ asyncio: {e}")

try:
    from fastapi import FastAPI
    print("   ✓ FastAPI")
except Exception as e:
    print(f"   ✗ FastAPI: {e}")

print("\n2. Testing mai_dx import (this might hang)...")
print("   If this hangs, the issue is in mai_dx module initialization")
sys.stdout.flush()  # Force output before potential hang

try:
    # This is likely where it hangs
    from mai_dx.main import MaiDxOrchestrator
    print("   ✓ mai_dx imported")
except Exception as e:
    print(f"   ✗ mai_dx: {e}")

print("\n✅ All imports successful!")
print("The server should work now.")
