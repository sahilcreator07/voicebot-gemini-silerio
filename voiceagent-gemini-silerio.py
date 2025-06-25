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

conversation_system_instruction = """You are a helpful chatbot who can search the web and get answers to user questions. You have access to MCP (Model Context Protocol) tools that allow you to search the internet and retrieve information.

When users ask questions, use the MCP tools available to you to search for relevant information and provide accurate, up-to-date answers. Always use the appropriate tools to find the information needed to help the user.

Guidelines:
- Use the MCP tools you have access to when users ask questions that require current information
- Search the web when needed to provide accurate and recent information
- Respond in clear, concise sentences suitable for audio playback
- Do not include special characters or formatting in your answers
- If the tools return multiple results, summarize or pick the most relevant information
- Be helpful and conversational in your responses
- If you cannot find information through the available tools, let the user know and suggest alternative approaches

Remember to always use the MCP tools at your disposal to provide the best possible assistance to users."""


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
            last_message = frame.context.messages[-1]
            self._context._messages.append(last_message)
            await self.push_frame(OpenAILLMContextFrame(context=self._context))


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

    # Register MCP tools
    try:
        if not os.getenv("FIRECRAWL_API_KEY"):
            logger.warning("FIRECRAWL_API_KEY not set. MCP tools will not be available.")
            mcp = None
            tools_schema = None
        else:
            if not shutil.which("npx"):
                logger.error("npx not found. Please install Node.js and npm.")
                mcp = None
                tools_schema = None
            else:
                mcp = MCPClient(
                    server_params=StdioServerParameters(
                        command=shutil.which("npx"),
                        args=["-y", "firecrawl-mcp"],
                        env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
                    )
                )
                tools_schema = await mcp.register_tools(conversation_llm)
                logger.info(f"Successfully registered {len(tools_schema.standard_tools)} MCP tools")
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")
        mcp = None
        tools_schema = None

    # Clean and convert tools to be compatible with Google Gemini
    cleaned_tools = []
    if tools_schema and tools_schema.standard_tools:
        for tool in tools_schema.standard_tools:
            # Create a clean version of the tool without problematic validation fields
            clean_tool = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": tool.required or []
                }
            }
            
            # Clean up properties to remove validation fields that Gemini doesn't accept
            if tool.properties:
                for prop_name, prop_def in tool.properties.items():
                    clean_prop = {
                        "type": prop_def.get("type", "string"),
                        "description": prop_def.get("description", "")
                    }
                    # Only add enum if it exists and is valid
                    if "enum" in prop_def:
                        clean_prop["enum"] = prop_def["enum"]
                    # If the property is an array, add items
                    if clean_prop["type"] == "array":
                        if "items" in prop_def:
                            clean_prop["items"] = prop_def["items"]
                        else:
                            clean_prop["items"] = {"type": "string"}
                    clean_tool["parameters"]["properties"][prop_name] = clean_prop
            
            cleaned_tools.append(clean_tool)
            logger.debug(f"Added tool: {tool.name}")
    
    # Create tools in the proper Google Gemini format
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

        await task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame(user_id="", timestamp=time.time(), text=message["message"]),
                UserStoppedSpeakingFrame(),
            ]
        )

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
