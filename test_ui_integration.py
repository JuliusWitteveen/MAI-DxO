#!/usr/bin/env python3
"""
Test script to verify that the UI integration is working properly.
This tests that the frontend correctly displays backend diagnostic data.
"""

import asyncio
import json
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mai_dx.main import MaiDxOrchestrator, CaseState, AgentRole


def test_backend_messages():
    """Test that backend generates expected UI messages"""
    print("🧪 Testing Backend Message Generation...")
    
    # Create orchestrator in interactive mode
    orchestrator = MaiDxOrchestrator(
        model_name="gpt-4o",
        max_iterations=3,
        mode="no_budget",
        interactive=True
    )
    
    # Test case from the paper
    initial_case = "A 29-year-old woman with sore throat and peritonsillar swelling"
    full_case = "29-year-old female with peritonsillar mass"
    ground_truth = "Rhabdomyosarcoma"
    
    print("✅ Orchestrator created successfully")
    
    # Test that run_gen is a generator
    gen = orchestrator.run_gen(initial_case, full_case, ground_truth)
    assert hasattr(gen, '__next__'), "run_gen should return a generator"
    print("✅ run_gen returns a generator")
    
    # Collect messages
    messages = []
    max_messages = 20
    
    try:
        for i in range(max_messages):
            try:
                msg = next(gen)
                messages.append(msg)
                
                # Print message types we receive
                if isinstance(msg, dict) and 'type' in msg:
                    print(f"📨 Message {i+1}: type='{msg['type']}'")
                    
                    # Check for expected message types
                    if msg['type'] == 'state_update':
                        print(f"   - Cost: ${msg.get('cumulative_cost', 0)}")
                        print(f"   - Iteration: {msg.get('iteration', 0)}")
                        if 'differential_diagnosis' in msg:
                            print(f"   - Diagnosis: {msg['differential_diagnosis'][:100]}...")
                    
                    elif msg['type'] == 'agent_status':
                        print(f"   - Agent: {msg.get('agent_id')}")
                        print(f"   - Status: {msg.get('status')}")
                    
                    elif msg['type'] == 'agent_update':
                        print(f"   - Agent: {msg.get('agent')}")
                        print(f"   - Content: {msg.get('content', '')[:50]}...")
                    
                    elif msg['type'] == 'pause':
                        print(f"   - Waiting for user input")
                        # Send dummy input to continue
                        gen.send("Normal physical exam except for the mass")
                        
            except StopIteration as e:
                # Generator completed
                result = e.value
                print(f"\n✅ Diagnosis complete: {getattr(result, 'final_diagnosis', 'Unknown')}")
                break
                
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    
    # Verify we got the expected message types
    message_types = {msg.get('type') for msg in messages if isinstance(msg, dict)}
    print(f"\n📊 Message types received: {message_types}")
    
    expected_types = {'state_update', 'agent_status', 'agent_update'}
    missing_types = expected_types - message_types
    
    if missing_types:
        print(f"⚠️  Missing message types: {missing_types}")
    else:
        print("✅ All expected message types received")
    
    return len(missing_types) == 0


def test_frontend_handlers():
    """Test that frontend has handlers for all message types"""
    print("\n🧪 Testing Frontend Message Handlers...")
    
    app_js_path = Path(__file__).parent / "webui" / "static" / "app.js"
    
    if not app_js_path.exists():
        print(f"❌ app.js not found at {app_js_path}")
        return False
    
    content = app_js_path.read_text()
    
    # Check for message handlers
    handlers = [
        ("state_update", "case 'state_update':"),
        ("agent_status", "case 'agent_status':"),
        ("agent_update", "case 'agent_update':"),
        ("pause", "case 'pause':"),
        ("result", "case 'result':"),
        ("log", "case 'log':"),
        ("done", "case 'done':")
    ]
    
    all_present = True
    for handler_name, handler_code in handlers:
        if handler_code in content:
            print(f"✅ Handler found for '{handler_name}'")
        else:
            print(f"❌ Missing handler for '{handler_name}'")
            all_present = False
    
    # Check for UI update functions
    functions = [
        ("updateDiagnosisList", "function updateDiagnosisList"),
        ("updateBudgetDisplay", "function updateBudgetDisplay"),
        ("updateRoundStatus", "function updateRoundStatus"),
        ("updateAgentInfo", "function updateAgentInfo"),
        ("updateFinalResult", "function updateFinalResult")
    ]
    
    print("\n🔍 Checking UI update functions...")
    for func_name, func_signature in functions:
        if func_signature in content:
            print(f"✅ Function found: {func_name}")
        else:
            print(f"❌ Missing function: {func_name}")
            all_present = False
    
    return all_present


def main():
    """Run all tests"""
    print("=" * 60)
    print("MAI-DxO UI Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Backend message generation
    backend_ok = test_backend_messages()
    
    # Test 2: Frontend handlers
    frontend_ok = test_frontend_handlers()
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("-" * 60)
    print(f"Backend Messages: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"Frontend Handlers: {'✅ PASS' if frontend_ok else '❌ FAIL'}")
    print("=" * 60)
    
    if backend_ok and frontend_ok:
        print("\n🎉 All tests passed! The UI should now properly display:")
        print("   • Real-time agent status updates")
        print("   • Differential diagnosis with probabilities")
        print("   • Cost tracking and budget indicators")
        print("   • Round/iteration counters")
        print("   • Agent reasoning and insights")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    return 0 if (backend_ok and frontend_ok) else 1


if __name__ == "__main__":
    exit(main())
