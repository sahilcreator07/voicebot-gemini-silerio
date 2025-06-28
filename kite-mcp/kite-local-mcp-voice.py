#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import time
import sys
import shlex
import shutil
import re
from mcp import StdioServerParameters
from pipecat.services.mcp_service import MCPClient
from google.genai.types import Content, Part
from dotenv import load_dotenv
from loguru import logger

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


def clean_markdown_for_voice(text):
    """Clean markdown formatting from text to make it suitable for voice output."""
    if not text:
        return text
    
    # Remove markdown formatting
    cleaned = text
    
    # Remove bold formatting (**text** or __text__)
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'__(.*?)__', r'\1', cleaned)
    
    # Remove italic formatting (*text* or _text_)
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
    cleaned = re.sub(r'_(.*?)_', r'\1', cleaned)
    
    # Remove code formatting (`text`)
    cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)
    
    # Remove URL formatting [text](url) -> text
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)
    
    # Remove bullet points and list markers
    cleaned = re.sub(r'^\s*[\*\-+]\s+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)
    
    # Remove headers (# ## ### etc)
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
    
    # Remove horizontal rules
    cleaned = re.sub(r'^[-*_]{3,}$', '', cleaned, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)
    
    return cleaned.strip()


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
Strictly Start the conversation with, "Hello! I am your Kite Assistant, your personal investment and trading companion. To get started, please log in to your Kite account when prompted."

Guidelines for Conversation:
- Do not read out unnecessary things like links, URLs, or special characters (e.g., &^%$#@!).
- Do not read out markdown formatting like **bold**, *italic*, or bullet points (*).
- Provide clean, natural speech without formatting characters.
- Focus on clear, relevant information only.
- Do not over-question the user; ask only what is necessary to assist them.
- Avoid repeating yourself or restating the same information multiple times.
- Always keep your responses concise, friendly, and easy to understand for voice.
- If you need the user to log in, politely prompt them once at the start.
- If you need to mention a link, say "I have sent a link to your screen" instead of reading the link aloud.
- Never read out API keys, tokens, or sensitive information.
- Respond in a conversational, helpful manner, and avoid complex jargon unless asked.
- If you do not know the answer, politely let the user know and offer to help with something else.
- Do not provide financial advice; always remind users to do their own research and consult professionals for investment decisions.

Capabilities:
- I can help you with real-time financial news, market updates, company information, and trading data through Kite and web search tools.
- I can analyze market trends, research companies, track stock prices, and provide investment insights.
- I will always use the latest data and tools to assist you.

Let's get started! Please log in to your Kite account to begin. How can I help you today?
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
            else:
                # Clean markdown formatting for voice output
                cleaned_text = clean_markdown_for_voice(frame.text)
                if cleaned_text != frame.text:
                    logger.debug(f"[SimpleOutputGate] Cleaned markdown: '{frame.text}' -> '{cleaned_text}'")
                    # Create a new TextFrame with cleaned text
                    frame = TextFrame(text=cleaned_text)

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

    # Register MCP tools from both Firecrawl and Kite
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

        # Local Kite MCP Server
        kite_server_path = os.path.join(os.path.dirname(__file__), "kite-server.py")
        if os.path.exists(kite_server_path):
            kite_mcp = MCPClient(
                server_params=StdioServerParameters(
                    command=sys.executable,  # Use current Python interpreter
                    args=[kite_server_path, "--mode", "stdio"],
                    env={
                        "ZERODHA_API_KEY": os.getenv("ZERODHA_API_KEY", ""),
                        "ZERODHA_API_SECRET": os.getenv("ZERODHA_API_SECRET", ""),
                        "ZERODHA_ACCESS_TOKEN": os.getenv("ZERODHA_ACCESS_TOKEN", ""),
                        "SERVER_MODE": "stdio",  # Ensure stdio mode
                    },
                )
            )
            kite_tools = await kite_mcp.register_tools(conversation_llm)
            mcp_clients.append((kite_mcp, kite_tools))
            logger.info(f"Registered {len(kite_tools.standard_tools)} Local Kite MCP tools.")
        else:
            kite_tools = None
            logger.warning(f"Local Kite MCP server not found at {kite_server_path}")

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
            # Register all MCP tool handlers with the LLM
            if hasattr(conversation_llm, "register_function_handler"):
                for mcp_client, tools in mcp_clients:
                    if tools and tools.standard_tools:
                        for tool in tools.standard_tools:
                            conversation_llm.register_function_handler(tool.name, mcp_client)
        else:
            logger.warning("No MCP tools available from any server.")
    except Exception as e:
        logger.error(f"Failed to initialize MCP clients: {e}")
        mcp_clients = []
        all_tools_schema = None

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

    # --- Detect Kite login tool name ---
    kite_login_tool_name = None
    kite_user_profile_tool_name = None
    if all_tools_schema and all_tools_schema.standard_tools:
        for tool in all_tools_schema.standard_tools:
            if tool.name == "get_login_url":
                kite_login_tool_name = tool.name
                logger.info(f"Detected Kite login tool: {kite_login_tool_name}")
            elif tool.name == "get_user_profile":
                kite_user_profile_tool_name = tool.name
                logger.info(f"Detected Kite user profile tool: {kite_user_profile_tool_name}")
            break

    async def check_kite_login_status():
        """Check if user is logged in to Kite by trying to get user profile."""
        if not kite_user_profile_tool_name:
            return False
        try:
            result = await conversation_llm.call_function(kite_user_profile_tool_name, {})
            return result is not None and result != "Error"
        except Exception as e:
            logger.debug(f"Kite login check failed: {e}")
            return False

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        
        # Check if user is logged in to Kite
        is_logged_in = await check_kite_login_status()
        
        if is_logged_in:
            await tts.queue_frame(TextFrame(text="Hello! I am your Kite Assistant, your personal investment and trading companion. You are logged in to your Kite account. How can I help you today?"))
        else:
            await tts.queue_frame(TextFrame(text="Hello! I am your Kite Assistant, your personal investment and trading companion. To access your portfolio and trading data, please log in to your Kite account. You can say 'login to kite' to get started. How can I help you today?"))

    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message):
        logger.debug(f"Received app message: {message}")
        if "message" not in message:
            return

        user_text = message["message"].lower()
        # If user asks to log in to Kite, call the login tool
        if kite_login_tool_name and ("login" in user_text and "kite" in user_text):
            await tts.queue_frame(TextFrame(text="Generating Kite login link..."))
            try:
                # Call the login tool via function calling
                result = await conversation_llm.call_function(kite_login_tool_name, {})
                # The local Kite server returns the URL directly as a string
                login_url = result if isinstance(result, str) else result.get("login_url") or result.get("url")
                if login_url:
                    await tts.queue_frame(TextFrame(text=f"Please log in to your Kite account using the link I sent to your screen."))
                    logger.info(f"Kite login URL: {login_url}")
                    print(f"Kite login URL: {login_url}")
                else:
                    await tts.queue_frame(TextFrame(text="Sorry, I couldn't get a login link from Kite."))
            except Exception as e:
                logger.error(f"Kite login tool call failed: {e}")
                await tts.queue_frame(TextFrame(text="Sorry, there was an error generating the Kite login link."))
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

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)

# Add debug logging to the output transport (SmallWebRTCOutputTransport)
orig_push_frame = SmallWebRTCOutputTransport.push_frame
async def debug_push_frame(self, frame, direction):
    logger.debug(f"[SmallWebRTCOutputTransport] Sending frame: {type(frame).__name__}, direction: {direction}")
    await orig_push_frame(self, frame, direction)
SmallWebRTCOutputTransport.push_frame = debug_push_frame
