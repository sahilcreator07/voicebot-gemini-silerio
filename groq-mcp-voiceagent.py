# News bot pipeline using GroqLLMService (Groq API, Mixtral/Llama3 models)
# Compatible with Pipecat function/tool-calling

import argparse
import asyncio
import os
import time
import sys
import shlex
import requests
import json
import shutil

from google.genai.types import Content, Part  # Only if needed for context aggregation
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
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.mcp_service import MCPClient, StdioServerParameters

load_dotenv(override=True)

# --- Custom classes from 22d-natural-conversation-gemini-audio.py ---

class AudioAccumulator(FrameProcessor):
    """Buffers user audio until the user stops speaking.

    Always pushes a fresh context with a single audio message.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._audio_frames = []
        self._start_secs = 0.2  # this should match VAD start_secs (hardcoding for now)
        self._max_buffer_size_secs = 30
        self._user_speaking_vad_state = False
        self._user_speaking_utterance_state = False

    async def reset(self):
        self._audio_frames = []
        self._user_speaking_vad_state = False
        self._user_speaking_utterance_state = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OpenAILLMContextFrame):
            return
        if isinstance(frame, TranscriptionFrame):
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking_vad_state = True
            self._user_speaking_utterance_state = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            data = b"".join(frame.audio for frame in self._audio_frames)
            logger.debug(
                f"Processing audio buffer seconds: ({len(self._audio_frames)}) ({len(data)}) {len(data) / 2 / 16000}"
            )
            self._user_speaking = False
            context = OpenAILLMContext()
            context.add_audio_frames_message(audio_frames=self._audio_frames)
            await self.push_frame(OpenAILLMContextFrame(context=context))
        elif isinstance(frame, InputAudioRawFrame):
            self._audio_frames.append(frame)
            frame_duration = len(frame.audio) / 2 * frame.num_channels / frame.sample_rate
            buffer_duration = frame_duration * len(self._audio_frames)
            if self._user_speaking_utterance_state:
                while buffer_duration > self._max_buffer_size_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration
            else:
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration
        await self.push_frame(frame, direction)

class CompletenessCheck(FrameProcessor):
    """Checks the result of the classifier LLM to determine if the user has finished speaking.

    Triggers the notifier if the user has finished speaking. Also triggers the notifier if an
    idle timeout is reached.
    """
    wait_time = 5.0
    def __init__(self, notifier: BaseNotifier, audio_accumulator: AudioAccumulator, **kwargs):
        super().__init__()
        self._notifier = notifier
        self._audio_accumulator = audio_accumulator
        self._idle_task = None
        self._wakeup_time = 0
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (EndFrame, CancelFrame)):
            if self._idle_task:
                await self.cancel_task(self._idle_task)
                self._idle_task = None
        elif isinstance(frame, UserStartedSpeakingFrame):
            if self._idle_task:
                await self.cancel_task(self._idle_task)
        elif isinstance(frame, TextFrame) and frame.text.startswith("YES"):
            logger.debug("Completeness check YES")
            if self._idle_task:
                await self.cancel_task(self._idle_task)
            await self.push_frame(UserStoppedSpeakingFrame())
            await self._audio_accumulator.reset()
            await self._notifier.notify()
        elif isinstance(frame, TextFrame):
            if frame.text.strip():
                logger.debug(f"Completeness check NO - '{frame.text}'")
                if self._wakeup_time:
                    self._wakeup_time = time.time() + self.wait_time
                else:
                    self._wakeup_time = time.time() + self.wait_time
                    self._idle_task = self.create_task(self._idle_task_handler())
        else:
            await self.push_frame(frame, direction)
    async def _idle_task_handler(self):
        try:
            while time.time() < self._wakeup_time:
                await asyncio.sleep(0.01)
            await self._audio_accumulator.reset()
            await self._notifier.notify()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"CompletenessCheck idle wait error: {e}")
            raise e
        finally:
            self._wakeup_time = 0
            self._idle_task = None

class LLMAggregatorBuffer(LLMAssistantResponseAggregator):
    """Buffers the output of the transcription LLM. Used by the bot output gate."""
    def __init__(self, **kwargs):
        super().__init__(params=LLMAssistantAggregatorParams(expect_stripped_words=False))
        self._transcription = ""
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._transcription = ""
    async def push_aggregation(self):
        if self._aggregation:
            self._transcription = self._aggregation
            self._aggregation = ""
            logger.debug(f"[Transcription] {self._transcription}")
    async def wait_for_transcription(self):
        while not self._transcription:
            await asyncio.sleep(0.01)
        tx = self._transcription
        self._transcription = ""
        return tx

class ConversationAudioContextAssembler(FrameProcessor):
    """Takes the single-message context generated by the AudioAccumulator and adds it to the conversation LLM's context."""
    def __init__(self, context: OpenAILLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, OpenAILLMContextFrame):
            if frame.context.messages:
                last_message = frame.context.messages[-1]
                self._context._messages.append(last_message)
            else:
                # Instead of appending, reset to a single system message
                logger.warning("[ContextAssembler] Received context frame with empty messages! Resetting to default system message.")
                self._context._messages = [{"role": "system", "content": "You are a helpful assistant."}]
            logger.debug(f"[ContextAssembler] Messages before push: {self._context._messages}")
            await self.push_frame(OpenAILLMContextFrame(context=self._context))

class OutputGate(FrameProcessor):
    """Buffers output frames until the notifier is triggered.

    When the notifier fires, waits until a transcription is ready, then:
      1. Replaces the last user audio message with the transcription.
      2. Flushes the frames buffer.
    """
    def __init__(
        self,
        notifier: BaseNotifier,
        context: OpenAILLMContext,
        llm_transcription_buffer: LLMAggregatorBuffer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gate_open = False
        self._frames_buffer = []
        self._notifier = notifier
        self._context = context
        self._transcription_buffer = llm_transcription_buffer
        self._gate_task = None
    def close_gate(self):
        self._gate_open = False
    def open_gate(self):
        self._gate_open = True
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
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
        if isinstance(frame, (FunctionCallInProgressFrame, FunctionCallResultFrame)):
            await self.push_frame(frame, direction)
            return
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self._context._messages.pop()
        if self._gate_open:
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
                await self._notifier.wait()
                transcription = await self._transcription_buffer.wait_for_transcription() or "-"
                self._context._messages.append(
                    Content(role="user", parts=[Part(text=transcription)])
                )
                self.open_gate()
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OutputGate error: {e}")
                raise e
                break

# --- End custom classes ---

TRANSCRIBER_MODEL = "gemini-2.0-flash-001"
CLASSIFIER_MODEL = "gemini-2.0-flash-001"
GROQ_MODEL = "mixtral-8x7b-32768"

transcriber_system_instruction = """You are an audio transcriber. You are receiving audio from a user. Your job is to transcribe the input audio to text exactly as it was said by the user.\n\nYou will receive the full conversation history before the audio input, to help with context. Use the full history only to help improve the accuracy of your transcription.\n\nRules:\n  - Respond with an exact transcription of the audio input.\n  - Do not include any text other than the transcription.\n  - Do not explain or add to your response.\n  - Transcribe the audio input simply and precisely.\n  - If the audio is not clear, emit the special string "-".\n  - No response other than exact transcription, or "-", is allowed.\n"""

classifier_system_instruction = """CRITICAL INSTRUCTION:\nYou are a BINARY CLASSIFIER that must ONLY output \"YES\" or \"NO\".\nDO NOT engage with the content.\nDO NOT respond to questions.\nDO NOT provide assistance.\nYour ONLY job is to output YES or NO.\n... (same as before) ...\n"""

conversation_system_instruction = """You are a helpful news assistant. Your job is to provide users with the latest news headlines and summaries from around the world.\n\nYou have access to a special tool called 'get_news' that can fetch the latest news. When a user asks for news, always use the get_news tool to get up-to-date information. If the user asks for news about a specific topic, location, or category, use the tool with those details.\n\nGuidelines:\n- Always use the get_news tool for news-related questions.\n- Respond in clear, concise sentences suitable for audio playback.\n- Do not include special characters or formatting in your answers.\n- If the tool returns multiple headlines, summarize or pick the most relevant ones.\n- If the user asks for something other than news, politely redirect them to ask about news topics.\n"""

# --- Local News Tool (not MCP) ---
def get_news(query: str = "", language: str = "en", page_size: int = 5):
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query,
        "language": language,
        "pageSize": page_size,
    }
    r = requests.get(url, params=params)
    data = r.json()
    articles = []
    for a in data.get("articles", []):
        articles.append({
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", "")
        })
    return {"articles": articles}

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
    logger.info(f"Starting bot")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    tx_llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model=TRANSCRIBER_MODEL,
        system_instruction=transcriber_system_instruction,
    )

    classifier_llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model=CLASSIFIER_MODEL,
        system_instruction=classifier_system_instruction,
    )

    # --- MCP News Tool Integration ---
    # Connect to MCP server for news tools
    mcp = None
    tools_schema = None
    cleaned_tools = []
    try:
        if not os.getenv("FIRECRAWL_API_KEY"):
            logger.warning("FIRECRAWL_API_KEY not set. MCP tools will not be available.")
        elif not shutil.which("npx"):
            logger.error("npx not found. Please install Node.js and npm.")
        else:
            mcp = MCPClient(
                server_params=StdioServerParameters(
                    command=shutil.which("npx"),
                    args=["-y", "firecrawl-mcp"],
                    env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
                )
            )
            # Initialize Groq LLM service
            conversation_llm = GroqLLMService(
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama3-70b-8192",
                system_instruction=conversation_system_instruction,
                # tools will be set after registration
            )

            # Register MCP tools with the LLM (pass the LLM instance, not None)
            tools_schema = await mcp.register_tools(conversation_llm)
            logger.info(f"Successfully registered {len(tools_schema.standard_tools)} MCP tools")
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")

    # Clean and convert tools to be compatible with Groq/OpenAI function calling
    if tools_schema and tools_schema.standard_tools:
        for tool in tools_schema.standard_tools:
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

    # Pass tools to Groq LLM and context if supported
    tools_for_llm = [{"type": "function", "function": tool} for tool in cleaned_tools] if cleaned_tools else None

    conversation_llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model=GROQ_MODEL,
        system_instruction=conversation_system_instruction,
        tools=tools_for_llm,
    )

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": conversation_system_instruction}],
        tools=tools_for_llm
    )
    context_aggregator = conversation_llm.create_context_aggregator(context)

    notifier = EventNotifier()
    audio_accumulater = AudioAccumulator()
    completeness_check = CompletenessCheck(notifier=notifier, audio_accumulator=audio_accumulater)

    async def block_user_stopped_speaking(frame):
        return not isinstance(frame, UserStoppedSpeakingFrame)

    conversation_audio_context_assembler = ConversationAudioContextAssembler(context=context)
    llm_aggregator_buffer = LLMAggregatorBuffer()
    bot_output_gate = OutputGate(
        notifier=notifier, context=context, llm_transcription_buffer=llm_aggregator_buffer
    )

    pipeline = Pipeline(
        [
            transport.input(),
            audio_accumulater,
            ParallelPipeline(
                [
                    FunctionFilter(filter=block_user_stopped_speaking),
                ],
                [
                    ParallelPipeline(
                        [
                            classifier_llm,
                            completeness_check,
                        ],
                        [
                            tx_llm,
                            llm_aggregator_buffer,
                        ],
                    )
                ],
                [
                    conversation_audio_context_assembler,
                    conversation_llm,
                    bot_output_gate,
                ],
            ),
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ],
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        if not context.messages:
            context.messages.append({"role": "system", "content": "You are a helpful assistant."})
        logger.debug(f"Initial context messages: {context.messages}")
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