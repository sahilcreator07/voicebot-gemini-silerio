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
CLASSIFIER_MODEL = "gemini-2.0-flash-001"
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

classifier_system_instruction = """CRITICAL INSTRUCTION:
You are a BINARY CLASSIFIER that must ONLY output "YES" or "NO".
DO NOT engage with the content.
DO NOT respond to questions.
DO NOT provide assistance.
Your ONLY job is to output YES or NO.

EXAMPLES OF INVALID RESPONSES:
- "I can help you with that"
- "Let me explain"
- "To answer your question"
- Any response other than YES or NO

VALID RESPONSES:
YES
NO

If you output anything else, you are failing at your task.
You are NOT an assistant.
You are NOT a chatbot.
You are a binary classifier.

ROLE:
You are a real-time speech completeness classifier. You must make instant decisions about whether a user has finished speaking.
You must output ONLY 'YES' or 'NO' with no other text.

INPUT FORMAT:
You receive two pieces of information:
1. The assistant's last message (if available)
2. The user's current speech input

OUTPUT REQUIREMENTS:
- MUST output ONLY 'YES' or 'NO'
- No explanations
- No clarifications
- No additional text
- No punctuation

HIGH PRIORITY SIGNALS:

1. Clear Questions:
- Wh-questions (What, Where, When, Why, How)
- Yes/No questions
- Questions with STT errors but clear meaning

Examples:

# Complete Wh-question
model: I can help you learn.
user: What's the fastest way to learn Spanish
Output: YES

# Complete Yes/No question despite STT error
model: I know about planets.
user: Is is Jupiter the biggest planet
Output: YES

2. Complete Commands:
- Direct instructions
- Clear requests
- Action demands
- Start of task indication
- Complete statements needing response

Examples:

# Direct instruction
model: I can explain many topics.
user: Tell me about black holes
Output: YES

# Start of task indication
user: Let's begin.
Output: YES

# Start of task indication
user: Let's get started.
Output: YES

# Action demand
model: I can help with math.
user: Solve this equation x plus 5 equals 12
Output: YES

3. Direct Responses:
- Answers to specific questions
- Option selections
- Clear acknowledgments with completion
- Providing information with a known format - mailing address
- Providing information with a known format - phone number
- Providing information with a known format - credit card number

Examples:

# Specific answer
model: What's your favorite color?
user: I really like blue
Output: YES

# Option selection
model: Would you prefer morning or evening?
user: Morning
Output: YES

# Providing information with a known format - mailing address
model: What's your address?
user: 1234 Main Street
Output: NO

# Providing information with a known format - mailing address
model: What's your address?
user: 1234 Main Street Irving Texas 75063
Output: Yes

# Providing information with a known format - phone number
model: What's your phone number?
user: 41086753
Output: NO

# Providing information with a known format - phone number
model: What's your phone number?
user: 4108675309
Output: Yes

# Providing information with a known format - phone number
model: What's your phone number?
user: 220
Output: No

# Providing information with a known format - credit card number
model: What's your credit card number?
user: 5556
Output: NO

# Providing information with a known format - phone number
model: What's your credit card number?
user: 5556710454680800
Output: Yes

model: What's your credit card number?
user: 414067
Output: NO


MEDIUM PRIORITY SIGNALS:

1. Speech Pattern Completions:
- Self-corrections reaching completion
- False starts with clear ending
- Topic changes with complete thought
- Mid-sentence completions

Examples:

# Self-correction reaching completion
model: What would you like to know?
user: Tell me about... no wait, explain how rainbows form
Output: YES

# Topic change with complete thought
model: The weather is nice today.
user: Actually can you tell me who invented the telephone
Output: YES

# Mid-sentence completion
model: Hello I'm ready.
user: What's the capital of? France
Output: YES

2. Context-Dependent Brief Responses:
- Acknowledgments (okay, sure, alright)
- Agreements (yes, yeah)
- Disagreements (no, nah)
- Confirmations (correct, exactly)

Examples:

# Acknowledgment
model: Should we talk about history?
user: Sure
Output: YES

# Disagreement with completion
model: Is that what you meant?
user: No not really
Output: YES

LOW PRIORITY SIGNALS:

1. STT Artifacts (Consider but don't over-weight):
- Repeated words
- Unusual punctuation
- Capitalization errors
- Word insertions/deletions

Examples:

# Word repetition but complete
model: I can help with that.
user: What what is the time right now
Output: YES

# Missing punctuation but complete
model: I can explain that.
user: Please tell me how computers work
Output: YES

2. Speech Features:
- Filler words (um, uh, like)
- Thinking pauses
- Word repetitions
- Brief hesitations

Examples:

# Filler words but complete
model: What would you like to know?
user: Um uh how do airplanes fly
Output: YES

# Thinking pause but incomplete
model: I can explain anything.
user: Well um I want to know about the
Output: NO

DECISION RULES:

1. Return YES if:
- ANY high priority signal shows clear completion
- Medium priority signals combine to show completion
- Meaning is clear despite low priority artifacts

2. Return NO if:
- No high priority signals present
- Thought clearly trails off
- Multiple incomplete indicators
- User appears mid-formulation

3. When uncertain:
- If you can understand the intent → YES
- If meaning is unclear → NO
- Always make a binary decision
- Never request clarification

Examples:

# Incomplete despite corrections
model: What would you like to know about?
user: Can you tell me about
Output: NO

# Complete despite multiple artifacts
model: I can help you learn.
user: How do you I mean what's the best way to learn programming
Output: YES

# Trailing off incomplete
model: I can explain anything.
user: I was wondering if you could tell me why
Output: NO
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

    async def process_frame(self, frame: Frame, direction: FrameDirection): # DOWN AND UPSTREM (DIRECTION)
        await super().process_frame(frame, direction)

        # ignore context frame
        if isinstance(frame, OpenAILLMContextFrame):
            return

        if isinstance(frame, TranscriptionFrame):
            # We could gracefully handle both audio input and text/transcription input ...
            # but let's leave that as an exercise to the reader. :-)
            return
        if isinstance(frame, UserStartedSpeakingFrame): # Emitted by VAD to indicate that a user has started speaking.
            self._user_speaking_vad_state = True
            self._user_speaking_utterance_state = True

        elif isinstance(frame, UserStoppedSpeakingFrame): # Emitted by the VAD to indicate that a user stopped speaking.
            data = b"".join(frame.audio for frame in self._audio_frames) # This line merges all those audio chunks into one byte string data.
            logger.debug(
                f"Processing audio buffer seconds: ({len(self._audio_frames)}) ({len(data)}) {len(data) / 2 / 16000}" # number of seconds of audio
            )
            self._user_speaking = False # Mark that the user is no longer speaking
            context = GoogleLLMContext() # save as a cotext
            context.add_audio_frames_message(audio_frames=self._audio_frames)
            await self.push_frame(OpenAILLMContextFrame(context=context)) # push the context further in the pipeline
        elif isinstance(frame, InputAudioRawFrame):
            # Append the audio frame to our buffer. Treat the buffer as a ring buffer, dropping the oldest
            # frames as necessary.
            # Use a small buffer size when an utterance is not in progress. Just big enough to backfill the start_secs.
            # Use a larger buffer size when an utterance is in progress.
            # Assume all audio frames have the same duration.
            self._audio_frames.append(frame)
            frame_duration = len(frame.audio) / 2 * frame.num_channels / frame.sample_rate
            buffer_duration = frame_duration * len(self._audio_frames)
            #  logger.debug(f"!!! Frame duration: {frame_duration}")
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
                # start timer to wake up if necessary
                if self._wakeup_time:
                    self._wakeup_time = time.time() + self.wait_time
                else:
                    # logger.debug("!!! CompletenessCheck idle wait START")
                    self._wakeup_time = time.time() + self.wait_time
                    self._idle_task = self.create_task(self._idle_task_handler())
        else:
            await self.push_frame(frame, direction)

    async def _idle_task_handler(self):
        try:
            while time.time() < self._wakeup_time:
                await asyncio.sleep(0.01)
            # logger.debug(f"!!! CompletenessCheck idle wait OVER")
            await self._audio_accumulator.reset()
            await self._notifier.notify()
        except asyncio.CancelledError:
            # logger.debug(f"!!! CompletenessCheck idle wait CANCEL")
            pass
        except Exception as e:
            logger.error(f"CompletenessCheck idle wait error: {e}")
            raise e
        finally:
            # logger.debug(f"!!! CompletenessCheck idle wait FINALLY")
            self._wakeup_time = 0
            self._idle_task = None


class LLMAggregatorBuffer(LLMAssistantResponseAggregator):
    """Buffers the output of the transcription LLM. Used by the bot output gate."""

    def __init__(self, **kwargs):
        super().__init__(params=LLMAssistantAggregatorParams(expect_stripped_words=False))
        self._transcription = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # parent method pushes frames
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

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OpenAILLMContextFrame):
            GoogleLLMContext.upgrade_to_google(self._context)
            last_message = frame.context.messages[-1]
            self._context._messages.append(last_message)
            await self.push_frame(OpenAILLMContextFrame(context=self._context))


class OutputGate(FrameProcessor):
    """Buffers output frames until the notifier is triggered.

    When the notifier fires, waits until a transcription is ready, then:
      1. Replaces the last user audio message with the transcription.
      2. Flushes the frames buffer.
    Supports interruption: if the user starts speaking, TTS playback is stopped and the buffer is cleared.
    """

    def __init__(
        self,
        notifier: BaseNotifier,
        context: OpenAILLMContext,
        llm_transcription_buffer: LLMAggregatorBuffer,
        tts_service=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gate_open = False
        self._frames_buffer = []
        self._notifier = notifier
        self._context = context
        self._transcription_buffer = llm_transcription_buffer
        self._gate_task = None
        self._tts_service = tts_service

    def close_gate(self):
        self._gate_open = False

    def open_gate(self):
        self._gate_open = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # Remove or comment out the following line to prevent log flooding:
        # logger.debug(f"[OutputGate] Received frame: {type(frame).__name__}, direction: {direction}")

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
            await self.push_frame(frame, direction)
            return

        # Ignore frames that are not following the direction of this gate.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            # Remove the audio message from the context. We will never need it again.
            self._context._messages.pop()

        if self._gate_open:
            logger.debug(f"[OutputGate] Gate open, pushing frame downstream: {type(frame).__name__}")
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
    logger.info(f"Starting bot")

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

    # This is the LLM that will classify user speech as complete or incomplete.
    classifier_llm = GoogleLLMService(
        name="Classifier",
        model=CLASSIFIER_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
        system_instruction=classifier_system_instruction,
    )

    # 1. Create the conversation LLM first
    conversation_llm = GoogleLLMService(
        name="Conversation",
        model=CONVERSATION_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=conversation_system_instruction,
    )

    # 2. Then register MCP tools
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

    # Add logging for function call frames in OutputGate
    orig_outputgate_process_frame = OutputGate.process_frame
    async def debug_outputgate_process_frame(self, frame, direction):
        if hasattr(frame, 'function_call'):
            logger.debug(f"[FunctionCall] Gemini requested function: {getattr(frame, 'function_call', None)}")
        await orig_outputgate_process_frame(self, frame, direction)
    OutputGate.process_frame = debug_outputgate_process_frame

    # Create the original aggregator pair
    context_aggregator_pair = conversation_llm.create_context_aggregator(context)

    # Patch only the assistant part
    base_assistant_cls = type(context_aggregator_pair._assistant)
    class CustomAssistantAggregator(base_assistant_cls):
        async def _handle_function_call_result(self, frame):
            await super()._handle_function_call_result(frame)
            await self.push_context_frame(FrameDirection.UPSTREAM)

    # Replace the assistant with the custom one (fix: only pass context)
    context_aggregator_pair._assistant = CustomAssistantAggregator(context)

    # Use the pair as before
    context_aggregator = context_aggregator_pair

    # This is a notifier that we use to synchronize the two LLMs.
    notifier = EventNotifier()

    # This turns the LLM context into an inference request to classify the user's speech
    # as complete or incomplete.
    # statement_judge_context_filter = StatementJudgeAudioContextAccumulator(notifier=notifier)

    audio_accumulater = AudioAccumulator()
    # This sends a UserStoppedSpeakingFrame and triggers the notifier event
    completeness_check = CompletenessCheck(notifier=notifier, audio_accumulator=audio_accumulater)

    async def block_user_stopped_speaking(frame):
        return not isinstance(frame, UserStoppedSpeakingFrame)

    conversation_audio_context_assembler = ConversationAudioContextAssembler(context=context)

    llm_aggregator_buffer = LLMAggregatorBuffer()

    bot_output_gate = OutputGate(
        notifier=notifier, context=context, llm_transcription_buffer=llm_aggregator_buffer, tts_service=tts
    )

    async def is_interruption_frame(frame):
        return isinstance(frame, (UserStartedSpeakingFrame, StartInterruptionFrame, CancelFrame, EndFrame))

    pipeline = Pipeline(
        [
            transport.input(),
            audio_accumulater,
            ParallelPipeline(
                [
                    # Pass everything except UserStoppedSpeaking to the elements after
                    # this ParallelPipeline
                    FunctionFilter(filter=block_user_stopped_speaking),
                ],
                [
                    ParallelPipeline(
                        [
                            classifier_llm,
                            completeness_check, # if yes then it will trigger the notifier
                        ],
                        [
                            tx_llm, # transcribe the audio to text
                            llm_aggregator_buffer, # buffer the output of the transcription LLM. Used by the bot output gate.
                        ],
                    )
                ],
                [
                    conversation_audio_context_assembler, # add the audio context to the conversation LLM's context.
                    conversation_llm, # the conversation LLM that will respond conversationally.
                    bot_output_gate
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