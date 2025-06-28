#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Required Environment Variables for Kite Connect:
# - KITE_API_KEY: Your Kite Connect API key from Zerodha developer console
# - KITE_API_SECRET: Your Kite Connect API secret from Zerodha developer console
# - GOOGLE_API_KEY: Google AI API key for LLM services
# - CARTESIA_API_KEY: Cartesia API key for TTS
# - FIRECRAWL_API_KEY: (Optional) Firecrawl API key for web search

import argparse
import asyncio
import os
import time
import sys
import shlex
import shutil
import webbrowser
from urllib.parse import urlparse, parse_qs
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import hashlib
import aiohttp
from mcp import StdioServerParameters
from pipecat.services.mcp_service import MCPClient
from google.genai.types import Content, Part
from dotenv import load_dotenv
from loguru import logger
import re

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    StartInterruptionFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantResponseAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMContext, GoogleLLMService
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams
from pipecat.transports.network.small_webrtc import SmallWebRTCOutputTransport

load_dotenv(override=True)


TRANSCRIBER_MODEL = "gemini-2.0-flash-001"
CONVERSATION_MODEL = "gemini-2.0-flash-001"

transcriber_system_instruction = """You are an audio transcriber. You are receiving audio from a user. Your job is to
transcribe the input audio to text exactly as it was said by the user.

You will receive the full conversation history before the audio input, to help with context. Use the full history only to help improve the accuracy of your transcription.

Rules:
  - Respond with an exact transcription of the audio input.
  - Do not include any text other than the transcription.
  - Do not explain or add to your response.
  - Transcribe the audio input simply and precisely.
  - If the audio is not clear, emit the special string "-".
  - No response other than exact transcription, or "-", is allowed.

"""

conversation_system_instruction = """
You are a personalized portfolio chatbot and investment advisor. You have access to powerful MCP (Model Context Protocol) tools that allow you to:

1. **Web Search & News Analysis**: Search the internet for real-time financial news, market updates, and company information (via Firecrawl MCP)
2. **Kite Trading Data**: Access real-time market data, stock prices, and trading information through Kite MCP server

Your capabilities include:
- Analyzing market trends and providing investment insights
- Researching companies and their financial performance
- Monitoring news that could impact investment decisions
- Providing personalized investment advice based on current market conditions
- Tracking stock prices and market movements
- Analyzing financial news for investment opportunities

Guidelines for Investment Advice:
- Always use your MCP tools to get the latest market data and news before making recommendations
- Consider market conditions, company fundamentals, and current events
- Provide balanced advice considering both opportunities and risks
- Be clear about the limitations of your advice (not financial advice, do your own research)
- Respond in clear, concise sentences suitable for voice conversation
- Avoid complex financial jargon unless necessary
- Always mention that users should consult with qualified financial advisors for personalized investment decisions

When users ask about investments:
1. Use web search to get the latest news and market information
2. Use Kite tools to get real-time market data when available
3. Analyze the information and provide insights
4. Give balanced recommendations considering both pros and cons
5. Always remind users to do their own research and consult professionals

Remember: Your output will be converted to audio, so keep responses conversational and easy to listen to. Always use your available tools to provide the most current and accurate information.
"""


class SimpleAudioAccumulator(FrameProcessor):
    """Simplified audio accumulator that works with Silero VAD.
    
    Buffers user audio until the user stops speaking, then creates a context frame.
    Much simpler than the original AudioAccumulator since VAD handles speech detection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._audio_frames = []
        self._max_buffer_size_secs = 30

    async def reset(self):
        self._audio_frames = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # ignore context frame
        if isinstance(frame, OpenAILLMContextFrame):
            return

        if isinstance(frame, TranscriptionFrame):
            # Handle text input directly
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            # User stopped speaking, create context from accumulated audio
            if self._audio_frames:
                data = b"".join(frame.audio for frame in self._audio_frames)
                logger.debug(
                    f"Processing audio buffer seconds: ({len(self._audio_frames)}) ({len(data)}) {len(data) / 2 / 16000}"
                )
                context = GoogleLLMContext()
                context.add_audio_frames_message(audio_frames=self._audio_frames)
                await self.push_frame(OpenAILLMContextFrame(context=context))
                self._audio_frames = []  # Reset buffer

        elif isinstance(frame, InputAudioRawFrame):
            # Append the audio frame to our buffer
            self._audio_frames.append(frame)
            
            # Manage buffer size to prevent memory issues
            frame_duration = len(frame.audio) / 2 * frame.num_channels / frame.sample_rate
            buffer_duration = frame_duration * len(self._audio_frames)
            
            while buffer_duration > self._max_buffer_size_secs:
                self._audio_frames.pop(0)
                buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


class ConversationAudioContextAssembler(FrameProcessor):
    """Takes the single-message context generated by the SimpleAudioAccumulator and adds it to the conversation LLM's context."""

    def __init__(self, context: OpenAILLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OpenAILLMContextFrame):
            GoogleLLMContext.upgrade_to_google(self._context)
            last_message = frame.context.messages[-1] # extract the last or most recent message from the context
            self._context._messages.append(last_message)
            await self.push_frame(OpenAILLMContextFrame(context=self._context)) # push the main memoryBuffer context frame downstream


class SimpleOutputGate(FrameProcessor):
    """Simplified output gate that works with Silero VAD.
    
    Buffers output frames until transcription is ready, then replaces the audio message
    with the transcription and flushes the buffer. Supports interruption via VAD.
    """

    def __init__(
        self,
        context: OpenAILLMContext,
        llm_transcription_buffer: LLMAssistantResponseAggregator,
        tts_service=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gate_open = False
        self._frames_buffer = []
        self._context = context
        self._transcription_buffer = llm_transcription_buffer
        self._gate_task = None
        self._tts_service = tts_service
        self._transcription_ready = False
        self._transcription_received = False

    def close_gate(self):
        self._gate_open = False

    def open_gate(self):
        self._gate_open = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            if isinstance(frame, StartFrame):
                await self._start()
            if isinstance(frame, (EndFrame, CancelFrame)):
                await self._stop()
            if isinstance(frame, StartInterruptionFrame):
                self._frames_buffer = []
                self.close_gate()
            await self.push_frame(frame, direction)
            return

        # Don't block function call frames
        if isinstance(frame, (FunctionCallInProgressFrame, FunctionCallResultFrame)):
            await self.push_frame(frame, direction)
            return

        # Interrupt TTS if user starts speaking
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._interrupt_tts()
            self._frames_buffer = []
            self.close_gate()
            self._transcription_received = False  # Reset for new conversation turn
            await self.push_frame(frame, direction)
            return

        # Handle transcription completion
        if isinstance(frame, TextFrame):
            # Debug: Log all TextFrame attributes to understand the structure
            logger.debug(f"[SimpleOutputGate] Received TextFrame: text='{frame.text}', source='{getattr(frame, 'source', 'NO_SOURCE')}', name='{getattr(frame, 'name', 'NO_NAME')}'")
            
            # Check if this is from the transcriber LLM
            # Since source is not set correctly, we'll check if this is the first TextFrame in a sequence
            # The transcriber typically produces the first TextFrame in a conversation turn
            if not hasattr(self, '_transcription_received'):
                self._transcription_received = False
            
            if not self._transcription_received:
                self._transcription_received = True
                self._transcription_ready = True
                transcription = frame.text.strip()
                logger.debug(f"[SimpleOutputGate] Received transcription: '{transcription}'")
                if transcription and transcription != "-":
                    # Replace the last audio message with transcription
                    if self._context._messages and self._context._messages[-1].role == "user":
                        self._context._messages[-1] = Content(role="user", parts=[Part(text=transcription)])
                    else:
                        self._context._messages.append(Content(role="user", parts=[Part(text=transcription)]))
                else:
                    logger.warning(f"[SimpleOutputGate] Transcriber returned '{transcription}', skipping response")
                    return  # Don't process this frame further

        # Ignore frames that are not following the direction of this gate.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            # Remove the audio message from the context. We will never need it again.
            if self._context._messages:
                self._context._messages.pop()

        if self._gate_open:
            logger.debug(f"[SimpleOutputGate] Gate open, pushing frame downstream: {type(frame).__name__}")
            await self.push_frame(frame, direction)
            return

        self._frames_buffer.append((frame, direction))

    async def _start(self):
        self._frames_buffer = []
        if not self._gate_task:
            self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

    async def _gate_task_handler(self):
        while True:
            try:
                # Wait for transcription to be ready
                while not self._transcription_ready:
                    await asyncio.sleep(0.01)
                
                self._transcription_ready = False
                self._transcription_received = False  # Reset for next conversation turn
                self.open_gate()
                
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SimpleOutputGate error: {e}")
                raise e

    async def _interrupt_tts(self):
        # If using CartesiaTTSService, call its stop method if available
        if self._tts_service and hasattr(self._tts_service, "stop"):
            try:
                await self._tts_service.stop()
                logger.info("TTS playback interrupted due to user speaking.")
            except Exception as e:
                logger.warning(f"Failed to interrupt TTS: {e}")


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


class KiteAuthHandler:
    def __init__(self, kite_mcp_client, conversation_llm, tts):
        self.kite_mcp_client = kite_mcp_client
        self.conversation_llm = conversation_llm
        self.tts = tts
        self.request_token = None
        self.auth_event = threading.Event()
        self.login_tool_name = None
        self.api_key = os.getenv("KITE_API_KEY")
        self.api_secret = os.getenv("KITE_API_SECRET")
        self.access_token = None
        self.user_id = None
        self.user_name = None
        self.last_auth_time = None
        self.callback_port = 8000
        self.callback_url = f"http://localhost:{self.callback_port}/callback"
        self.start_callback_server()
        self.load_session()

    def start_callback_server(self):
        def make_handler(*args, **kwargs):
            return self.CallbackHandler(self, *args, **kwargs)
        self.auth_server = HTTPServer(('localhost', self.callback_port), make_handler)
        server_thread = threading.Thread(target=self.auth_server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        logger.info(f"Started always-on callback server on port {self.callback_port}")

    def find_login_tool(self, tools_schema):
        """Find the Kite login tool name"""
        if tools_schema and tools_schema.standard_tools:
            for tool in tools_schema.standard_tools:
                if "login" in tool.name.lower() and ("kite" in tool.name.lower() or tool.name.startswith("kite:")):
                    self.login_tool_name = tool.name
                    logger.info(f"Found Kite login tool: {self.login_tool_name}")
                    return tool.name
        return None
    
    def get_free_port(self):
        """Get a free port for the callback server"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    class CallbackHandler(BaseHTTPRequestHandler):
        def __init__(self, auth_handler, *args, **kwargs):
            self.auth_handler = auth_handler
            super().__init__(*args, **kwargs)
        
        def do_GET(self):
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            if 'request_token' in query_params:
                self.auth_handler.request_token = query_params['request_token'][0]
                self.auth_handler.auth_event.set()
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'''
                <html>
                <body>
                <h2>Authentication Successful!</h2>
                <p>You can now close this window and return to your bot.</p>
                <script>window.close();</script>
                </body>
                </html>
                ''')
            else:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<html><body><h2>Authentication Failed</h2></body></html>')
        
        def log_message(self, format, *args):
            # Suppress default logging
            pass
    
    def generate_checksum(self, request_token):
        """Generate SHA-256 checksum as required by Kite Connect"""
        if not self.api_key or not self.api_secret:
            raise ValueError("KITE_API_KEY and KITE_API_SECRET must be set in environment variables")
        
        checksum_string = f"{self.api_key}{request_token}{self.api_secret}"
        return hashlib.sha256(checksum_string.encode()).hexdigest()
    
    async def exchange_token(self, request_token):
        """Exchange request_token for access_token via Kite Connect API"""
        try:
            checksum = self.generate_checksum(request_token)
            
            # Prepare the token exchange request
            data = {
                "api_key": self.api_key,
                "request_token": request_token,
                "checksum": checksum
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.kite.trade/session/token",
                    headers={"X-Kite-Version": "3"},
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("status") == "success":
                            data = result.get("data", {})
                            self.access_token = data.get("access_token")
                            self.user_id = data.get("user_id")
                            self.user_name = data.get("user_name")
                            self.last_auth_time = time.time()
                            
                            logger.info(f"Token exchange successful for user: {self.user_name}")
                            return True
                        else:
                            logger.error(f"Token exchange failed: {result}")
                            return False
                    else:
                        logger.error(f"Token exchange HTTP error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error during token exchange: {e}")
            return False
    
    async def is_session_valid(self):
        """Check if the current session is still valid"""
        if not self.access_token or not self.api_key:
            return False
            
        try:
            # Try to get user profile to verify session
            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-Kite-Version": "3",
                    "Authorization": f"token {self.api_key}:{self.access_token}"
                }
                
                async with session.get("https://api.kite.trade/user/profile", headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("status") == "success":
                            return True
                    elif response.status == 401:
                        logger.info("Session expired (401 Unauthorized)")
                        return False
                    else:
                        logger.warning(f"Session validation returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.debug(f"Session validation failed: {e}")
        
        return False
    
    async def handle_kite_login(self):
        """Handle the complete Kite login flow"""
        try:
            if await self.is_session_valid():
                await self.tts.queue_frame(TextFrame(text=f"You're already logged in to Kite as {self.user_name}. Your session is active."))
                return True
            if not self.api_key or not self.api_secret:
                await self.tts.queue_frame(TextFrame(text="Kite API credentials not configured. Please set KITE_API_KEY and KITE_API_SECRET environment variables."))
                return False
            # Use the always-on callback server and URL
            callback_url = self.callback_url
            await self.tts.queue_frame(TextFrame(text="Starting Kite login process..."))
            login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"
            redirect_params = f"callback_url=https://a594-115-98-232-130.ngrok-free.app/callback"
            login_url += f"&redirect_params={redirect_params}"
            await self.tts.queue_frame(TextFrame(text="Opening your browser for Kite authentication. Please complete the login process."))
            logger.info(f"Kite login URL: {login_url}")
            print(f"\n=== KITE LOGIN URL ===")
            print(f"{login_url}")
            print(f"======================\n")
            try:
                webbrowser.open(login_url)
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
            await self.tts.queue_frame(TextFrame(text="Waiting for you to complete the authentication..."))
            auth_received = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.auth_event.wait(timeout=60)
            )
            if not auth_received or not self.request_token:
                await self.tts.queue_frame(TextFrame(text="Authentication timed out. Please try again."))
                return False
            await self.tts.queue_frame(TextFrame(text="Completing authentication..."))
            if await self.exchange_token(self.request_token):
                self.save_session()
                await self.tts.queue_frame(TextFrame(text=f"Successfully logged in to your Kite account as {self.user_name}! I can now access your trading data."))
                return True
            else:
                await self.tts.queue_frame(TextFrame(text="Authentication failed. Please check your credentials and try again."))
                return False
        except Exception as e:
            logger.error(f"Error during Kite login: {e}")
            await self.tts.queue_frame(TextFrame(text="There was an error during the login process. Please try again."))
            return False
        finally:
            self.auth_event.clear()
            self.request_token = None
    
    async def refresh_session_if_needed(self):
        """Refresh the session if it's expired or about to expire"""
        if not await self.is_session_valid():
            logger.info("Kite session expired, refreshing...")
            self.access_token = None  # Clear expired token
            return await self.handle_kite_login()
        return True
    
    async def call_kite_tool_with_session_management(self, tool_name, arguments=None):
        """Call a Kite tool with automatic session management"""
        if arguments is None:
            arguments = {}
        
        try:
            # First, ensure we have a valid session
            if not await self.is_session_valid():
                logger.info(f"Session invalid for {tool_name}, attempting to refresh...")
                if not await self.refresh_session_if_needed():
                    return {"error": "Failed to authenticate with Kite"}
            
            # Now call the tool via the MCP client
            if self.kite_mcp_client:
                result = await self.kite_mcp_client.call_tool(tool_name, arguments)
                
                # Check if the result indicates session expiration
                if result and isinstance(result, str) and "Please log in first" in result:
                    logger.info(f"Session expired during {tool_name} call, refreshing...")
                    self.access_token = None  # Clear expired token
                    if await self.refresh_session_if_needed():
                        # Retry the call once
                        result = await self.kite_mcp_client.call_tool(tool_name, arguments)
                    else:
                        return {"error": "Failed to refresh Kite session"}
                
                return result
            else:
                # Fallback to LLM function calling
                result = await self.conversation_llm.call_function(tool_name, arguments)
                
                # Check if the result indicates session expiration
                if result and isinstance(result, str) and "Please log in first" in result:
                    logger.info(f"Session expired during {tool_name} call, refreshing...")
                    self.access_token = None  # Clear expired token
                    if await self.refresh_session_if_needed():
                        # Retry the call once
                        result = await self.conversation_llm.call_function(tool_name, arguments)
                    else:
                        return {"error": "Failed to refresh Kite session"}
                
                return result
            
        except Exception as e:
            logger.error(f"Error calling Kite tool {tool_name}: {e}")
            return {"error": f"Failed to call {tool_name}: {str(e)}"}
    
    def save_session(self):
        """Save the current session to a file"""
        if self.access_token and self.last_auth_time:
            session_data = {
                "access_token": self.access_token,
                "user_id": self.user_id,
                "user_name": self.user_name,
                "last_auth_time": self.last_auth_time,
                "api_key": self.api_key  # Store for convenience
            }
            try:
                with open("kite_session.json", "w") as f:
                    json.dump(session_data, f)
                logger.info("Kite session saved")
            except Exception as e:
                logger.error(f"Failed to save session: {e}")
    
    def load_session(self):
        """Load the session from file"""
        try:
            if os.path.exists("kite_session.json"):
                with open("kite_session.json", "r") as f:
                    session_data = json.load(f)
                self.access_token = session_data.get("access_token")
                self.user_id = session_data.get("user_id")
                self.user_name = session_data.get("user_name")
                self.last_auth_time = session_data.get("last_auth_time")
                logger.info(f"Kite session loaded for user: {self.user_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
        return False
    
    def clear_session(self):
        """Clear the current session"""
        self.access_token = None
        self.user_id = None
        self.user_name = None
        self.last_auth_time = None
        try:
            if os.path.exists("kite_session.json"):
                os.remove("kite_session.json")
                logger.info("Kite session file removed")
        except Exception as e:
            logger.error(f"Failed to remove session file: {e}")


# --- GLOBAL MCP CLIENTS ---
# Create MCP clients ONCE at startup and reuse them
firecrawl_mcp = None
kite_mcp = None
kite_session = None  # Persistent session for Kite
mcp_clients = []
all_tools_schema = None

async def initialize_mcp_clients(conversation_llm):
    global firecrawl_mcp, kite_mcp, kite_session, mcp_clients, all_tools_schema
    mcp_clients = []
    all_tools_schema = None
    try:
        # Firecrawl MCP
        if os.getenv("FIRECRAWL_API_KEY") and shutil.which("npx"):
            firecrawl_mcp = MCPClient(
                server_params=StdioServerParameters(
                    command=shutil.which("npx"),
                    args=["-y", "firecrawl-mcp"],
                    env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
                )
            )
            firecrawl_tools = await firecrawl_mcp.register_tools(conversation_llm)
            mcp_clients.append((firecrawl_mcp, firecrawl_tools))
            logger.info(f"Registered {len(firecrawl_tools.standard_tools)} Firecrawl MCP tools.")
        else:
            firecrawl_tools = None
            logger.warning("Firecrawl MCP not available.")

        # Kite MCP - Use standard MCP client approach
        if shutil.which("npx"):
            logger.info("[MCP SESSION] Creating Kite MCP client...")
            try:
                kite_mcp = MCPClient(
                    server_params=StdioServerParameters(
                        command=shutil.which("npx"),
                        args=["-y", "kite-mcp"],
                    )
                )
                kite_tools = await kite_mcp.register_tools(conversation_llm)
                mcp_clients.append((kite_mcp, kite_tools))
                logger.info(f"[MCP SESSION] Created Kite MCP client with {len(kite_tools.standard_tools)} tools")
            except Exception as e:
                logger.error(f"Failed to create Kite MCP client: {e}")
                kite_mcp = None
                kite_tools = None
        else:
            kite_tools = None
            logger.warning("Kite MCP not available.")

        # Combine all tools
        all_standard_tools = []
        if firecrawl_tools and firecrawl_tools.standard_tools:
            all_standard_tools.extend(firecrawl_tools.standard_tools)
        if kite_tools and kite_tools.standard_tools:
            all_standard_tools.extend(kite_tools.standard_tools)
        if all_standard_tools:
            from pipecat.services.mcp_service import ToolsSchema
            all_tools_schema = ToolsSchema(standard_tools=all_standard_tools)
            logger.info(f"Combined {len(all_standard_tools)} total MCP tools from all servers.")
            
            # Register function handlers with the LLM
            if hasattr(conversation_llm, "register_function_handler"):
                for mcp_client, tools in mcp_clients:
                    if tools and tools.standard_tools:
                        for tool in tools.standard_tools:
                            # Create function handlers that use the appropriate MCP client
                            async def create_mcp_handler(tool_name, mcp_client):
                                async def mcp_handler(function_name, tool_call_id, arguments, llm, context, result_callback):
                                    try:
                                        result = await mcp_client.call_tool(tool_name, arguments or {})
                                        return result
                                    except Exception as e:
                                        logger.error(f"Error calling {tool_name}: {e}")
                                        return {"error": f"Failed to call {tool_name}: {str(e)}"}
                                return mcp_handler
                            
                            handler = await create_mcp_handler(tool.name, mcp_client)
                            conversation_llm.register_function_handler(tool.name, handler)
                            logger.debug(f"Registered tool '{tool.name}' with {mcp_client}")
        else:
            logger.warning("No MCP tools available from any server.")
    except Exception as e:
        logger.error(f"Failed to initialize MCP clients: {e}")
        mcp_clients = []
        all_tools_schema = None


async def cleanup_mcp_clients():
    """Clean up MCP clients and sessions"""
    global kite_session
    if kite_session:
        try:
            await kite_session.close()
            logger.info("[MCP SESSION] Closed persistent Kite MCP session")
        except Exception as e:
            logger.error(f"Error closing Kite MCP session: {e}")
        finally:
            kite_session = None


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot with Silero VAD")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # This is the LLM that will transcribe user speech.
    tx_llm = GoogleLLMService(
        name="Transcriber",
        model=TRANSCRIBER_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
        system_instruction=transcriber_system_instruction,
    )

    # Create the conversation LLM
    conversation_llm = GoogleLLMService(
        name="Conversation",
        model=CONVERSATION_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=conversation_system_instruction,
    )

    # Add error handling for rate limits
    @conversation_llm.event_handler("on_error")
    async def on_conversation_error(service, error):
        if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
            logger.warning("Google API rate limit hit, waiting before retry...")
            await tts.queue_frame(TextFrame(text="I'm experiencing high demand right now. Please wait a moment and try again."))
        else:
            logger.error(f"Conversation LLM error: {error}")
            await tts.queue_frame(TextFrame(text="I encountered an error. Please try again."))

    @tx_llm.event_handler("on_error")
    async def on_transcriber_error(service, error):
        if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
            logger.warning("Google API rate limit hit for transcriber, waiting before retry...")
            # Don't speak for transcriber errors to avoid spam
        else:
            logger.error(f"Transcriber LLM error: {error}")

    # Register MCP tools from both Firecrawl and Kite
    await initialize_mcp_clients(conversation_llm)

    # Clean and convert tools to be compatible with Google Gemini
    cleaned_tools = []
    if all_tools_schema and all_tools_schema.standard_tools:
        for tool in all_tools_schema.standard_tools:
            clean_tool = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": tool.required or []
                }
            }
            if tool.properties:
                for prop_name, prop_def in tool.properties.items():
                    clean_prop = {
                        "type": prop_def.get("type", "string"),
                        "description": prop_def.get("description", "")
                    }
                    if "enum" in prop_def:
                        clean_prop["enum"] = prop_def["enum"]
                    if clean_prop["type"] == "array":
                        if "items" in prop_def:
                            clean_prop["items"] = prop_def["items"]
                        else:
                            clean_prop["items"] = {"type": "string"}
                    clean_tool["parameters"]["properties"][prop_name] = clean_prop
            cleaned_tools.append(clean_tool)
            logger.debug(f"Added tool: {tool.name}")
    gemini_tools = [
        {
            "function_declarations": cleaned_tools
        }
    ] if cleaned_tools else None
    logger.debug(f"[GeminiTools] Tools passed to Gemini: {gemini_tools}")

    context = OpenAILLMContext(tools=gemini_tools)

    # Create the context aggregator pair
    context_aggregator_pair = conversation_llm.create_context_aggregator(context)

    # Patch only the assistant part to handle function calls properly
    base_assistant_cls = type(context_aggregator_pair._assistant)
    class CustomAssistantAggregator(base_assistant_cls):
        async def _handle_function_call_result(self, frame):
            await super()._handle_function_call_result(frame)
            await self.push_context_frame(FrameDirection.UPSTREAM)

    # Replace the assistant with the custom one
    context_aggregator_pair._assistant = CustomAssistantAggregator(context)

    # Use the pair as before
    context_aggregator = context_aggregator_pair

    # Create simplified components
    simple_audio_accumulator = SimpleAudioAccumulator()
    conversation_audio_context_assembler = ConversationAudioContextAssembler(context=context)
    
    # Create transcription buffer
    llm_aggregator_buffer = LLMAssistantResponseAggregator(
        params=LLMAssistantAggregatorParams(expect_stripped_words=False)
    )

    # Create simplified output gate
    simple_output_gate = SimpleOutputGate(
        context=context, 
        llm_transcription_buffer=llm_aggregator_buffer, 
        tts_service=tts
    )

    # Create pipeline with Silero VAD handling interruptions
    pipeline = Pipeline([
        transport.input(),  # Silero VAD handles speech detection and interruptions
        simple_audio_accumulator,
        ParallelPipeline([
            tx_llm,  # Transcribe audio to text
            llm_aggregator_buffer,  # Buffer transcription results
        ], [
            conversation_audio_context_assembler,  # Add audio context to conversation
            conversation_llm,  # Generate response
            simple_output_gate,  # Gate output until transcription is ready
        ]),
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # Enable interruptions via Silero VAD
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Initialize Kite auth handler
    kite_auth_handler = KiteAuthHandler(
        kite_mcp_client=kite_mcp,  # Use the standard kite_mcp client
        conversation_llm=conversation_llm,
        tts=tts
    )

    # Try to load existing session
    if kite_auth_handler.load_session():
        logger.info("Loaded existing Kite session")
    else:
        logger.info("No existing Kite session found")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message):
        logger.debug(f"Received app message: {message}")
        if "message" not in message:
            return

        user_text = message["message"].lower()
        
        # Handle explicit login requests first
        if any(phrase in user_text for phrase in ["login", "connect", "authorize", "sign in"]):
            if kite_mcp:
                try:
                    logger.debug(f"[MCP SESSION] Explicit login request detected, calling login function")
                    result = await kite_mcp.call_tool("login", {})
                    logger.debug(f"[MCP SESSION] Login result: {result}")
                    
                    if isinstance(result, str) and "https://kite.zerodha.com/connect/login" in result:
                        # Extract the login URL
                        login_url = extract_login_url(result)
                        if login_url:
                            await tts.queue_frame(TextFrame(text="I'm generating a login link for you. Please click on the link that will appear in your browser to authorize access to your Kite account."))
                            print(f"\n=== KITE LOGIN LINK ===")
                            print(f"{login_url}")
                            print(f"========================\n")
                            try:
                                webbrowser.open(login_url)
                            except Exception as e:
                                logger.warning(f"Could not open browser automatically: {e}")
                            await tts.queue_frame(TextFrame(text="The login link has been opened in your browser. Please complete the authorization process and let me know when you're done."))
                        else:
                            await tts.queue_frame(TextFrame(text="I've generated a login link. Please check your browser or the console output for the authorization link."))
                    else:
                        await tts.queue_frame(TextFrame(text="I encountered an issue generating the login link. Please try again."))
                except Exception as e:
                    logger.error(f"Error calling login function: {e}")
                    await tts.queue_frame(TextFrame(text="Sorry, I encountered an error while trying to generate the login link."))
            else:
                await tts.queue_frame(TextFrame(text="Kite MCP client is not available. Please check your setup."))
            return
        
        # Handle Kite login requests
        if kite_auth_handler.login_tool_name:
            # Check for explicit login requests
            if ("login" in user_text and "kite" in user_text) or ("connect" in user_text and "kite" in user_text):
                success = await kite_auth_handler.handle_kite_login()
                if success:
                    # Optionally fetch some initial data after login
                    try:
                        profile_result = await kite_auth_handler.call_kite_tool_with_session_management("kite:get_profile", {})
                        if profile_result and not profile_result.get('error'):
                            user_name = profile_result.get("user_name", kite_auth_handler.user_name or "User")
                            await tts.queue_frame(TextFrame(text=f"Welcome {user_name}! Your Kite account is now connected and ready for trading queries."))
                    except Exception as e:
                        logger.debug(f"Could not fetch profile after login: {e}")
                return
            
            # Check for logout/clear session requests
            if any(phrase in user_text for phrase in ["logout", "disconnect", "clear session", "forget me", "sign out"]):
                kite_auth_handler.clear_session()
                await tts.queue_frame(TextFrame(text="I've cleared your Kite session. You'll need to log in again next time."))
                return
            
            # Check for session status requests
            if any(phrase in user_text for phrase in ["session status", "login status", "am i logged in", "check login"]):
                if await kite_auth_handler.is_session_valid():
                    user_name = kite_auth_handler.user_name or "User"
                    await tts.queue_frame(TextFrame(text=f"You are logged in to Kite as {user_name}. Your session is active."))
                else:
                    await tts.queue_frame(TextFrame(text="You are not currently logged in to Kite. Say 'login to kite' to connect your account."))
                return

        # Example: handle Kite holdings request using standard MCP client
        if "holdings" in user_text or "portfolio" in user_text:
            if kite_mcp:
                try:
                    logger.debug(f"[MCP SESSION] Calling get_holdings with standard kite_mcp client")
                    result = await kite_mcp.call_tool("get_holdings", {})
                    logger.debug(f"[MCP SESSION] get_holdings result: {result}")
                    
                    # Check if login is required
                    if isinstance(result, str) and ("login" in result.lower() or "authorize" in result.lower()):
                        login_url = extract_login_url(result)
                        if login_url:
                            await tts.queue_frame(TextFrame(text="Please log in to Kite using the link I just sent you. I will wait for you to complete authentication."))
                            print(f"\nKite Login Link: {login_url}\n")
                            try:
                                webbrowser.open(login_url)
                            except Exception:
                                pass
                        else:
                            await tts.queue_frame(TextFrame(text="Please log in to Kite using the link provided in your chat window."))
                            print(f"\nKite Login Message: {result}\n")
                        await tts.queue_frame(TextFrame(text="Let me know when you have completed the login. I will retry your request in a few seconds."))
                        logger.debug("[MCP SESSION] Waiting 15 seconds for session propagation after login...")
                        await asyncio.sleep(15)  # Wait for user to complete login and session to propagate
                        result = await kite_mcp.call_tool("get_holdings", {})
                        logger.debug(f"[MCP SESSION] get_holdings result after login: {result}")
                    
                    if isinstance(result, dict) and result.get("error"):
                        await tts.queue_frame(TextFrame(text="Sorry, I was unable to retrieve your holdings."))
                    else:
                        await tts.queue_frame(TextFrame(text=f"Here are your current holdings: {result}"))
                except Exception as e:
                    logger.error(f"Error calling get_holdings: {e}")
                    await tts.queue_frame(TextFrame(text="Sorry, I encountered an error while retrieving your holdings."))
            else:
                await tts.queue_frame(TextFrame(text="Kite MCP client is not available. Please check your setup."))
            return

        # Debug: Test session persistence
        if "test session" in user_text or "check session" in user_text:
            if kite_mcp:
                try:
                    logger.debug(f"[MCP SESSION] Testing session persistence with get_profile")
                    result = await kite_mcp.call_tool("get_profile", {})
                    logger.debug(f"[MCP SESSION] get_profile result: {result}")
                    
                    if isinstance(result, str) and ("login" in result.lower() or "authorize" in result.lower()):
                        await tts.queue_frame(TextFrame(text="Session test failed - login required. The session is not persisting."))
                    else:
                        await tts.queue_frame(TextFrame(text="Session test successful! The session is persisting correctly."))
                except Exception as e:
                    logger.error(f"Error testing session: {e}")
                    await tts.queue_frame(TextFrame(text=f"Session test failed with error: {e}"))
            else:
                await tts.queue_frame(TextFrame(text="Kite MCP client is not available for session testing."))
            return

        await task.queue_frames([
                UserStartedSpeakingFrame(),
                TranscriptionFrame(user_id="", timestamp=time.time(), text=message["message"]),
                UserStoppedSpeakingFrame(),
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    try:
        await runner.run(task)
    finally:
        # Clean up MCP sessions
        await cleanup_mcp_clients()


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)

# Add debug logging to the output transport (SmallWebRTCOutputTransport)
orig_push_frame = SmallWebRTCOutputTransport.push_frame
async def debug_push_frame(self, frame, direction):
    logger.debug(f"[SmallWebRTCOutputTransport] Sending frame: {type(frame).__name__}, direction: {direction}")
    await orig_push_frame(self, frame, direction)
SmallWebRTCOutputTransport.push_frame = debug_push_frame

def extract_login_url(response_text):
    match = re.search(r'(https://kite\.zerodha\.com/connect/login\?[^)\s]+)', response_text)
    return match.group(1) if match else None
