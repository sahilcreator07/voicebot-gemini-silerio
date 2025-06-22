# Natural Conversation with Gemini Audio - Code Documentation

## Overview

This document explains the `22d-natural-conversation-gemini-audio.py` file, which implements a sophisticated voice conversation system using multiple AI models working together in real-time.

## System Architecture

The system creates a natural voice conversation flow:
1. **User speaks** → Audio is captured
2. **Speech is transcribed** → Text is generated  
3. **System determines if user finished speaking** → Completeness check
4. **AI responds conversationally** → Response is generated
5. **Response is converted to speech** → Audio is played back

## Key Components

### 1. AI Models and Services

#### Three Google Gemini Models
```python
TRANSCRIBER_MODEL = "gemini-2.0-flash-001"    # Converts audio to text
CLASSIFIER_MODEL = "gemini-2.0-flash-001"     # Determines if speech is complete
CONVERSATION_MODEL = "gemini-2.0-flash-001"   # Generates conversational responses
```

#### Text-to-Speech Service
```python
tts = CartesiaTTSService(
    api_key=os.getenv("CARTESIA_API_KEY"),
    voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
)
```

### 2. System Instructions (AI Prompts)

#### Transcriber AI
```python
transcriber_system_instruction = """You are an audio transcriber..."""
```
- **Purpose**: Converts audio to text
- **Rules**: Must be exact transcription, returns "-" if unclear
- **No explanations or additions allowed**

#### Classifier AI  
```python
classifier_system_instruction = """CRITICAL INSTRUCTION: You are a BINARY CLASSIFIER..."""
```
- **Purpose**: Determines if user finished speaking
- **Output**: Only "YES" or "NO"
- **Uses sophisticated rules** to detect speech completion

#### Conversation AI
```python
conversation_system_instruction = """You are a helpful assistant..."""
```
- **Purpose**: Generates natural conversational responses
- **Style**: Concise responses suitable for audio output

## Core Classes Explained

### AudioAccumulator
```python
class AudioAccumulator(FrameProcessor):
```

**Purpose**: Buffers user audio until the user stops speaking.

**Key Features**:
- Collects audio frames during speech
- Manages buffer size dynamically:
  - **30 seconds max** when user is speaking
  - **0.2 seconds** when user is not speaking (for quick start detection)
- Creates context with audio data for AI processing

**How it works**:
```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    if isinstance(frame, UserStartedSpeakingFrame):
        # User started speaking - begin collecting audio
        self._user_speaking_utterance_state = True
    
    elif isinstance(frame, UserStoppedSpeakingFrame):
        # User stopped speaking - process collected audio
        context = GoogleLLMContext()
        context.add_audio_frames_message(audio_frames=self._audio_frames)
        await self.push_frame(OpenAILLMContextFrame(context=context))
    
    elif isinstance(frame, InputAudioRawFrame):
        # Add audio frame to buffer, manage buffer size
        self._audio_frames.append(frame)
        # Remove old frames if buffer too large
```

### CompletenessCheck
```python
class CompletenessCheck(FrameProcessor):
```

**Purpose**: Determines when user finishes speaking and triggers processing.

**Key Features**:
- Receives "YES"/"NO" from classifier AI
- If "YES" → triggers notifier to process response
- If "NO" → waits with 5-second timeout
- Manages idle timeout for incomplete speech

**How it works**:
```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    if isinstance(frame, TextFrame) and frame.text.startswith("YES"):
        # Speech is complete - trigger processing
        await self.push_frame(UserStoppedSpeakingFrame())
        await self._audio_accumulator.reset()
        await self._notifier.notify()
    elif isinstance(frame, TextFrame):
        # Speech incomplete - start timeout timer
        if not self._idle_task:
            self._idle_task = self.create_task(self._idle_task_handler())
```

### LLMAggregatorBuffer
```python
class LLMAggregatorBuffer(LLMAssistantResponseAggregator):
```

**Purpose**: Manages transcription results and provides synchronization.

**Why we need it**:
- **Timing Coordination**: Ensures conversation waits for complete transcription
- **Quality Assurance**: Only uses final, complete transcriptions
- **Pipeline Synchronization**: Coordinates multiple parallel processes

**How it works**:
```python
async def wait_for_transcription(self):
    while not self._transcription:
        await asyncio.sleep(0.01)  # Wait until transcription is ready
    tx = self._transcription
    self._transcription = ""  # Clear for next use
    return tx
```

**Real-world analogy**: Like a waiter ensuring food is fully cooked before serving.

### ConversationAudioContextAssembler
```python
class ConversationAudioContextAssembler(FrameProcessor):
```

**Purpose**: Takes audio context and adds it to conversation history.

**How it works**:
```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    if isinstance(frame, OpenAILLMContextFrame):
        # Add the audio message to conversation context
        last_message = frame.context.messages[-1]
        self._context._messages.append(last_message)
        await self.push_frame(OpenAILLMContextFrame(context=self._context))
```

### OutputGate
```python
class OutputGate(FrameProcessor):
```

**Purpose**: Controls when to process responses - buffers output until user finishes speaking.

**Key Features**:
- Buffers all output frames until user finishes speaking
- Waits for transcription to be ready
- Replaces audio message with transcription text
- Flushes buffered responses when ready

**How it works**:
```python
async def _gate_task_handler(self):
    while True:
        await self._notifier.wait()  # Wait for speech completion
        
        # Get transcription and add to context
        transcription = await self._transcription_buffer.wait_for_transcription()
        self._context._messages.append(
            Content(role="user", parts=[Part(text=transcription)])
        )
        
        # Open gate and flush buffered frames
        self.open_gate()
        for frame, direction in self._frames_buffer:
            await self.push_frame(frame, direction)
        self._frames_buffer = []
```

## Pipeline Architecture

The system uses a sophisticated parallel pipeline:

```
User Audio → AudioAccumulator → Parallel Processing:
                                    ├── Classifier LLM → CompletenessCheck
                                    ├── Transcriber LLM → LLMAggregatorBuffer  
                                    └── ConversationAudioContextAssembler → Conversation LLM → OutputGate → TTS → Audio Output
```

### Pipeline Breakdown:
```python
pipeline = Pipeline([
    transport.input(),                    # Audio input from user
    audio_accumulater,                    # Buffer audio during speech
    ParallelPipeline([
        # Branch 1: Filter out UserStoppedSpeaking frames
        FunctionFilter(filter=block_user_stopped_speaking),
    ], [
        # Branch 2: Parallel processing
        ParallelPipeline([
            # Path A: Speech completeness detection
            classifier_llm,               # Classify if speech is complete
            completeness_check,           # Trigger processing if complete
        ], [
            # Path B: Transcription
            tx_llm,                       # Transcribe audio to text
            llm_aggregator_buffer,        # Buffer transcription results
        ], [
            # Path C: Conversation
            conversation_audio_context_assembler,  # Add audio to context
            conversation_llm,             # Generate conversational response
            bot_output_gate,              # Buffer output until ready
        ]),
    ]),
    tts,                                  # Convert response to speech
    transport.output(),                   # Audio output to user
    context_aggregator.assistant(),       # Update conversation context
])
```

## Speech Completeness Detection

The classifier uses sophisticated rules to determine if someone finished speaking:

### High Priority Signals:
- **Clear questions**: What, Where, When, Why, How
- **Complete commands**: "Tell me about black holes"
- **Direct responses**: "I really like blue"

### Medium Priority Signals:
- **Self-corrections reaching completion**: "Tell me about... no wait, explain how rainbows form"
- **Topic changes with complete thoughts**: "Actually can you tell me who invented the telephone"

### Low Priority Signals:
- **Filler words**: um, uh, like
- **Speech artifacts**: transcription errors, repeated words

## Synchronization Mechanism

The system uses an `EventNotifier` to coordinate between:
1. **Speech completeness detection**
2. **Transcription completion**  
3. **Response generation and output**

```python
notifier = EventNotifier()

# CompletenessCheck triggers the notifier
await self._notifier.notify()

# OutputGate waits for the notifier
await self._notifier.wait()
```

## Transport Options

The system supports multiple transport methods:

```python
transport_params = {
    "daily": lambda: DailyParams(...),    # Video conferencing
    "twilio": lambda: FastAPIWebsocketParams(...),  # Phone calls  
    "webrtc": lambda: TransportParams(...),  # Web browser (default)
}
```

## Event Handlers

### Client Connection
```python
@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    # Start conversation when client connects
    await task.queue_frames([context_aggregator.user().get_context_frame()])
```

### Text Messages (for testing)
```python
@transport.event_handler("on_app_message")
async def on_app_message(transport, message):
    # Handle text messages (useful for testing without microphone)
    await task.queue_frames([
        UserStartedSpeakingFrame(),
        TranscriptionFrame(text=message["message"]),
        UserStoppedSpeakingFrame(),
    ])
```

### Client Disconnection
```python
@transport.event_handler("on_client_disconnected")
async def on_client_disconnected(transport, client):
    # Clean up when client disconnects
    await task.cancel()
```

## Key Innovation: Speech Completeness Detection

The most sophisticated part is the classifier that determines if someone finished speaking. It uses a comprehensive set of rules to make binary decisions:

### Decision Rules:
1. **Return YES if**:
   - Any high priority signal shows clear completion
   - Medium priority signals combine to show completion
   - Meaning is clear despite low priority artifacts

2. **Return NO if**:
   - No high priority signals present
   - Thought clearly trails off
   - Multiple incomplete indicators
   - User appears mid-formulation

3. **When uncertain**:
   - If you can understand the intent → YES
   - If meaning is unclear → NO
   - Always make a binary decision

## Error Handling

The code includes robust error handling for:
- **Audio processing errors**
- **AI service failures** 
- **Network disconnections**
- **Timeout scenarios**
- **Cancellation events**

## Usage Examples

### Running with WebRTC (local testing):
```bash
python 22d-natural-conversation-gemini-audio.py --transport webrtc
```

### Running with Daily (video calls):
```bash
python 22d-natural-conversation-gemini-audio.py --transport daily
```

### Running with Twilio (phone calls):
```bash
python 22d-natural-conversation-gemini-audio.py --transport twilio --proxy your-domain.com
```

## Environment Variables Required

```env
GOOGLE_API_KEY=your_google_api_key_here
CARTESIA_API_KEY=your_cartesia_api_key_here
```

## Summary

This system creates a sophisticated, real-time voice conversation experience by:

1. **Coordinating multiple AI models** working in parallel
2. **Using intelligent speech detection** to determine when users finish speaking
3. **Synchronizing transcription and conversation** for accurate responses
4. **Buffering and timing** to ensure smooth, natural interactions
5. **Supporting multiple transport methods** for different use cases

The result is a voice agent that feels natural, responsive, and intelligent in conversation. 