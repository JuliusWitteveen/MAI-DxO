import pytest
from .main import MaiDxOrchestrator

# This is the data structure from the log that caused the original code to fail.
# It simulates the response from the swarms agent.
MOCK_AGENT_RESPONSE_FROM_LOG = [
    {"role": "System", "content": "Your Name: Consensus Coordinator"},
    {"role": "Human", "content": "[TOKEN MANAGEMENT...]"},
    {
        "role": "Consensus Coordinator",
        "content": [
            {
                "function": {
                    "arguments": '{"action_type":"test","content":["Complete Blood Count (CBC) with Differential and Peripheral Blood Smear","Monospot Test (Heterophile Antibody Test)"],"reasoning":"The panel\'s analysis strongly supports..."}',
                    "name": "make_consensus_decision"
                },
                "id": "call_V1txPbEQ68lu3uTWXN4I2tua",
                "type": "function"
            }
        ]
    }
]

def test_parser_handles_swarms_conversation_list_format():
    """
    Asserts that the patched parser can handle the nested list format.
    This test will PASS with your corrected code, but would have FAILED with the original.
    """
    # Create an instance of the orchestrator to access its internal methods
    orchestrator = MaiDxOrchestrator()
    
    # Call the parser function with the problematic data
    result = orchestrator._extract_function_call_output(MOCK_AGENT_RESPONSE_FROM_LOG)

    # Check that the result is correct
    assert result is not None, "Parser incorrectly returned None."
    assert isinstance(result, dict), "Parser did not return a dictionary."
    assert result.get("action_type") == "test", "Parser failed to extract 'action_type'."
    assert "content" in result, "Parser failed to extract 'content'."
    assert len(result["content"]) == 2, "Parser extracted the wrong number of tests."