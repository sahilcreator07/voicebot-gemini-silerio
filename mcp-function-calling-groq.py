#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import shutil

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams
from mcp import StdioServerParameters
from pipecat.services.mcp_service import MCPClient

load_dotenv(override=True)


conversation_system_instruction = """
You are a helpful AI assistant in a voice conversation. 

You have access to tools that can help you search for news and information (if MCP tools are available).

Guidelines:
- Keep responses concise and natural for voice conversation
- Use the available tools when appropriate
- After using a tool, always provide a clear summary of the information found
- Avoid special characters or formatting that would sound unnatural when spoken
- Be helpful and engaging
- If you cannot find information, let the user know politely

Remember: Your output will be converted to audio, so keep it conversational and easy to listen to.
"""
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

    stt = GroqSTTService(api_key=os.getenv("GROQ_API_KEY"), model="distil-whisper-large-v3-en")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-4-maverick-17b-128e-instruct"
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        logger.info(f"Function calls started: {[f.function_name for f in function_calls]}")
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    # Register MCP tools first
    mcp_tools_schema = None
    try:
        if not os.getenv("FIRECRAWL_API_KEY"):
            logger.warning("FIRECRAWL_API_KEY not set. MCP tools will not be available.")
            mcp = None
        else:
            if not shutil.which("npx"):
                logger.error("npx not found. Please install Node.js and npm.")
                mcp = None
            else:
                mcp = MCPClient(
                    server_params=StdioServerParameters(
                        command=shutil.which("npx"),
                        args=["-y", "firecrawl-mcp"],
                        env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
                    )
                )
                # Register MCP tools with the LLM
                mcp_tools_schema = await mcp.register_tools(llm)
                logger.info(f"Successfully registered {len(mcp_tools_schema.standard_tools)} MCP tools")
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")
        mcp = None
        mcp_tools_schema = None

    # Create tools schema with available tools
    available_tools = []
    
    # Add MCP tools if available
    if mcp_tools_schema and mcp_tools_schema.standard_tools:
        available_tools.extend(mcp_tools_schema.standard_tools)
        logger.info(f"Added {len(mcp_tools_schema.standard_tools)} MCP tools to schema")
    
    tools = ToolsSchema(standard_tools=available_tools)
    messages = [
        {
            "role": "system",
            "content": conversation_system_instruction,
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(
        context, user_params=LLMUserAggregatorParams(aggregation_timeout=0.05)
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
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
        logger.info("Sending initial context frame")
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        logger.info("Initial context frame sent")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
