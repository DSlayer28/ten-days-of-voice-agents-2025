import logging

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any
from livekit.plugins import murf

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentTask,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")



# ...

@dataclass
class Concept:
    id: str
    title: str
    summary: str
    sample_question: str

def load_concepts() -> Dict[str, Concept]:
    content_path = Path(__file__).parent.parent / "shared data" / "day4_tutor_content.json"
    with content_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    concepts = {item["id"]: Concept(**item) for item in items}
    return concepts


class TutorAgentBase(Agent):
    def __init__(self, chat_ctx, concepts, concept_id, instructions, tts):
        # pass chat_ctx and tts to Agent so session and voice are wired correctly
        super().__init__(chat_ctx=chat_ctx, instructions=instructions, tts=tts)
        # DO NOT assign self.chat_ctx directly — Agent provides a read-only property
        self._concepts = concepts
        self._current = concept_id
        # keep tts reference if you want, but Agent already stores it
        self._tts = tts   # optional private copy



    @property
    def concept(self):
        return self._concepts[self._current]

    @function_tool()
    async def change_concept(self, context: RunContext, concept_id: str):
        if concept_id not in self._concepts:
            await self.session.generate_reply(instructions=f"Concept '{concept_id}' not available. Options: {', '.join(self._concepts.keys())}")
            return
        self._current = concept_id
        await self.session.generate_reply(instructions=f"Switched to {self.concept.title}.")
    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."

class LearnAgent(TutorAgentBase):
    def __init__(self, chat_ctx, concepts, concept_id):
        tts = murf.TTS(voice="en-US-matthew", style="Conversation", tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2), text_pacing=True)
        super().__init__(chat_ctx, concepts, concept_id,
            instructions=(
                "LEARN mode: Explain the current concept clearly using the summary. Give one short example. Ask if user wants quiz, teach-back, or another example."
            ),
            tts=tts)

    async def on_enter(self):
        c = self.concept
        await self.session.generate_reply(instructions=f"Explain '{c.title}': {c.summary}\nThen offer an example and ask if they'd like quiz or teach-back.")

class QuizAgent(TutorAgentBase):
    def __init__(self, chat_ctx, concepts, concept_id):
        tts = murf.TTS(voice="en-US-alicia", style="Conversation", tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2), text_pacing=True)
        super().__init__(chat_ctx, concepts, concept_id,
            instructions=("QUIZ mode: Ask the sample question and up to 2 short follow-ups based on the user's answer. Provide brief feedback."),
            tts=tts)

    async def on_enter(self):
        c = self.concept
        await self.session.generate_reply(instructions=f"Quiz: {c.sample_question}\nWait for answer then ask follow-up.")


class TeachBackAgent(TutorAgentBase):
    def __init__(self, chat_ctx, concepts, concept_id):
        tts = murf.TTS(voice="en-US-ken", style="Conversation", tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2), text_pacing=True)
        super().__init__(chat_ctx, concepts, concept_id,
            instructions=("TEACH_BACK mode: Ask the user to teach the concept back. After they speak, give 2-sentence qualitative feedback: 1 thing correct, 1 suggestion."),
            tts=tts)

    async def on_enter(self):
        c = self.concept
        await self.session.generate_reply(instructions=f"Please explain the concept '{c.title}' to me in your own words.")

    

class RouterAgent(Agent):
    def __init__(self, concepts):
        super().__init__(instructions="Router: Greet user, explain modes (learn/quiz/teach_back), ask concept choice. Use tools to hand off.")
        self._concepts = concepts

    async def on_enter(self):
        await self.session.generate_reply(instructions="Hi — choose a mode (learn/quiz/teach_back) and a topic (e.g. variables or loops).")

    def _validate(self, cid: str):
        if cid not in self._concepts:
            raise ValueError(f"Unknown concept {cid}")

    @function_tool()
    async def go_to_learn(self, context: RunContext, concept_id: str):
        self._validate(concept_id)
        agent = LearnAgent(chat_ctx=self.chat_ctx, concepts=self._concepts, concept_id=concept_id)
        return agent, f"Switching to Learn mode for {concept_id}"

    @function_tool()
    async def go_to_quiz(self, context: RunContext, concept_id: str):
        self._validate(concept_id)
        agent = QuizAgent(chat_ctx=self.chat_ctx, concepts=self._concepts, concept_id=concept_id)
        return agent, f"Switching to Quiz mode for {concept_id}"

    @function_tool()
    async def go_to_teach_back(self, context: RunContext, concept_id: str):
        self._validate(concept_id)
        agent = TeachBackAgent(chat_ctx=self.chat_ctx, concepts=self._concepts, concept_id=concept_id)
        return agent, f"Switching to Teach-Back mode for {concept_id}"



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    concepts = load_concepts()
    await session.start(
        agent=RouterAgent(concepts=concepts),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )


    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

    print(Path(__file__).parent.parent.resolve())
    print((Path(__file__).parent.parent / "shared data" / "day4_tutor_content.json").exists())
