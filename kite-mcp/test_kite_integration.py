#!/usr/bin/env python3
"""
Test script to verify the local Kite MCP server integration.
"""

import asyncio
import os
import sys
from mcp import StdioServerParameters
from pipecat.services.mcp_service import MCPClient
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

async def test_kite_mcp_integration():
    """Test the local Kite MCP server integration."""
    
    # Check if kite-server.py exists
    kite_server_path = os.path.join(os.path.dirname(__file__), "kite-server.py")
    if not os.path.exists(kite_server_path):
        logger.error(f"Kite server not found at {kite_server_path}")
        return False
    
    # Check environment variables
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("ZERODHA_API_KEY and ZERODHA_API_SECRET must be set in .env file")
        return False
    
    try:
        # Create MCP client for local Kite server
        kite_mcp = MCPClient(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=[kite_server_path, "--mode", "stdio"],
                env={
                    "ZERODHA_API_KEY": api_key,
                    "ZERODHA_API_SECRET": api_secret,
                    "SERVER_MODE": "stdio",  # Ensure stdio mode
                },
            )
        )
        
        # Test connection and tool registration
        logger.info("Testing Kite MCP server connection...")
        
        # Create a dummy LLM service for testing
        class DummyLLM:
            def __init__(self):
                self.name = "test"
        
        dummy_llm = DummyLLM()
        
        # Register tools
        tools = await kite_mcp.register_tools(dummy_llm)
        
        if not tools or not tools.standard_tools:
            logger.error("No tools registered from Kite MCP server")
            return False
        
        logger.info(f"Successfully registered {len(tools.standard_tools)} Kite MCP tools:")
        
        # List available tools
        for tool in tools.standard_tools:
            logger.info(f"  - {tool.name}: {tool.description}")
        
        # Test the get_login_url tool
        login_tool = None
        for tool in tools.standard_tools:
            if tool.name == "get_login_url":
                login_tool = tool
                break
        
        if login_tool:
            logger.info("Testing get_login_url tool...")
            try:
                # Call the tool
                result = await kite_mcp.call_tool("get_login_url", {})
                if result and isinstance(result, str) and "kite.trade" in result:
                    logger.info(f"✅ Login URL generated successfully: {result[:50]}...")
                else:
                    logger.error(f"❌ Unexpected result from get_login_url: {result}")
                    return False
            except Exception as e:
                logger.error(f"❌ Error calling get_login_url: {e}")
                return False
        else:
            logger.warning("get_login_url tool not found")
        
        logger.info("✅ Kite MCP integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Kite MCP integration test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting Kite MCP integration test...")
    
    success = await test_kite_mcp_integration()
    
    if success:
        logger.info("🎉 All tests passed! The Kite MCP integration is working correctly.")
        sys.exit(0)
    else:
        logger.error("💥 Tests failed! Please check the configuration and try again.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 