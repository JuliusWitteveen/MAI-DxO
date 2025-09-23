"""
Windows compatibility layer for swarms framework.
This module provides shims and workarounds for Windows-specific issues.
"""

import sys
import types
from typing import Any, Dict, Optional
import warnings

def install_windows_shims():
    """Install compatibility shims for Windows platform."""
    
    # 1. uvloop shim (not available on Windows)
    if "uvloop" not in sys.modules:
        uvloop = types.ModuleType("uvloop")
        
        def install():
            return None
        
        class EventLoopPolicy:
            pass
        
        def new_event_loop_policy():
            return EventLoopPolicy()
        
        uvloop.install = install
        uvloop.EventLoopPolicy = EventLoopPolicy
        uvloop.new_event_loop_policy = new_event_loop_policy
        
        sys.modules["uvloop"] = uvloop
    
    # 2. MCP (Model Context Protocol) shims - check for missing modules
    try:
        import mcp.types
        # Check if streamable_http is available
        try:
            import mcp.client.streamable_http
            # MCP is properly installed with all modules, no need for shims
            return
        except ImportError:
            # MCP is installed but missing streamable_http, create shims for missing parts
            pass
    except ImportError:
        # MCP not available at all, create full shims
        pass
    
    # Handle case where MCP is partially installed (missing streamable_http)
    if "mcp.client.streamable_http" not in sys.modules:
        # Create just the missing streamable_http module
        mcp_client_streamable = types.ModuleType("mcp.client.streamable_http")
        
        def streamablehttp_client(*args, **kwargs):
            """Dummy streamable HTTP client."""
            return None
        
        mcp_client_streamable.streamablehttp_client = streamablehttp_client
        sys.modules["mcp.client.streamable_http"] = mcp_client_streamable
        
        # Log that we're using compatibility mode for missing modules
        warnings.warn(
            "MCP streamable_http module not available. Using compatibility shim. Some advanced tool features may not work.",
            RuntimeWarning,
            stacklevel=2
        )
        return
    
    if "mcp" not in sys.modules:
        # Root mcp module
        mcp = types.ModuleType("mcp")
        
        # mcp.types module
        mcp_types = types.ModuleType("mcp.types")
        
        # Define all the types that swarms expects
        class ClientSession:
            def __init__(self, *args, **kwargs):
                pass
        
        class CallToolRequestParams:
            def __init__(self, *args, **kwargs):
                pass
        
        class CallToolResult:
            def __init__(self, *args, **kwargs):
                pass
        
        class TextContent:
            def __init__(self, text: str):
                self.text = text
        
        class ImageContent:
            def __init__(self, *args, **kwargs):
                pass
        
        class EmbeddedResourceContent:
            def __init__(self, *args, **kwargs):
                pass
        
        class CreateMessageRequest:
            def __init__(self, *args, **kwargs):
                pass
        
        class CreateMessageResult:
            def __init__(self, *args, **kwargs):
                pass
        
        class Empty:
            def __init__(self, *args, **kwargs):
                pass
        
        class InitializeRequest:
            def __init__(self, *args, **kwargs):
                pass
        
        class InitializeResult:
            def __init__(self, *args, **kwargs):
                pass
        
        class ListToolsRequest:
            def __init__(self, *args, **kwargs):
                pass
        
        class ListToolsResult:
            def __init__(self, *args, **kwargs):
                self.tools = []
        
        class Tool:
            def __init__(self, name: str, description: str, schema: Optional[Dict] = None):
                self.name = name
                self.description = description
                self.schema = schema or {}
        
        # Assign types to mcp.types
        mcp_types.ClientSession = ClientSession
        mcp_types.CallToolRequestParams = CallToolRequestParams
        mcp_types.CallToolResult = CallToolResult
        mcp_types.TextContent = TextContent
        mcp_types.ImageContent = ImageContent
        mcp_types.EmbeddedResourceContent = EmbeddedResourceContent
        mcp_types.CreateMessageRequest = CreateMessageRequest
        mcp_types.CreateMessageResult = CreateMessageResult
        mcp_types.Empty = Empty
        mcp_types.InitializeRequest = InitializeRequest
        mcp_types.InitializeResult = InitializeResult
        mcp_types.ListToolsRequest = ListToolsRequest
        mcp_types.ListToolsResult = ListToolsResult
        mcp_types.Tool = Tool
        
        # mcp.client module
        mcp_client = types.ModuleType("mcp.client")
        
        # mcp.client.streamable_http module
        mcp_client_streamable = types.ModuleType("mcp.client.streamable_http")
        
        def streamablehttp_client(*args, **kwargs):
            """Dummy streamable HTTP client."""
            return None
        
        mcp_client_streamable.streamablehttp_client = streamablehttp_client
        
        # Register the streamable_http module
        sys.modules["mcp.client.streamable_http"] = mcp_client_streamable
        
        # Wire up the module hierarchy
        mcp.types = mcp_types
        mcp.client = mcp_client
        mcp.ClientSession = ClientSession  # Also expose at root level
        mcp_client.streamable_http = mcp_client_streamable
        
        # Register all modules
        sys.modules["mcp"] = mcp
        sys.modules["mcp.types"] = mcp_types
        sys.modules["mcp.client"] = mcp_client
        sys.modules["mcp.client.streamable_http"] = mcp_client_streamable
        
        # Log that we're using compatibility mode
        warnings.warn(
            "MCP (Model Context Protocol) is not available on Windows. "
            "Using compatibility shims. Some advanced tool features may not work.",
            RuntimeWarning,
            stacklevel=2
        )


def patch_swarms_imports():
    """Apply any necessary patches to swarms imports."""
    # This function can be extended if we need to patch specific swarms behaviors
    pass


def ensure_windows_compatibility():
    """Main entry point to ensure Windows compatibility."""
    if sys.platform == "win32":
        install_windows_shims()
        patch_swarms_imports()
