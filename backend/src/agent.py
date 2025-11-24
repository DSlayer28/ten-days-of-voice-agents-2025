import logging

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from livekit.agents import Agent, function_tool, RunContext
from livekit.agents import get_job_context  # if you actually use this elsewhere



from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

@dataclass
class WellnessCheckin:
    mood: str
    stress: str
    sleepHours: float
    activityMinutes: int
    focusArea: str
    reflection: str



class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a calm, supportive health & wellness voice companion.\n"
                "Your goal is to guide the user through a short check-in and save their responses.\n"
                "You MUST use these tools to store information: set_mood, set_stress, set_sleep, "
                "set_activity, set_focus_area, set_reflection.\n"
                "Collect these fields: mood, stress, sleepHours, activityMinutes, focusArea, reflection.\n"
                "Ask one short question at a time (1 to 2 sentences) and keep your tone practical and non-dramatic.\n"
                "Do NOT diagnose or give medical treatment. For serious issues, tell the user to seek a doctor or local emergency help.\n"
                "Flow: ask how they feel → stress → last night's sleep → today's activity → main focus area → brief reflection.\n"
                "Use tools as soon as a detail is clear. When all fields are stored, summarize their day in 2 to 3 sentences and end the check-in.\n"
            )
        )
        # internal partial state
        self._order: dict[str, object] = {}

    async def _check_completion(self) -> None:
        required = {"mood", "stress", "sleepHours", "activityMinutes", "focusArea", "reflection"}

        if required.issubset(self._state.keys()):
            checkin = WellnessCheckin(
                mood=str(self._state["mood"]),
                stress=str(self._state["stress"]),
                sleepHours=float(self._state["sleepHours"]),
                activityMinutes=int(self._state["activityMinutes"]),
                focusArea=str(self._state["focusArea"]),
                reflection=str(self._state["reflection"]),
            )
            await self._on_checkin_complete(checkin)
        else:
            await self.session.generate_reply(
                instructions="Continue the wellness check-in and ask about any missing details."
            )

    async def _on_checkin_complete(self, checkin: WellnessCheckin) -> None:
        # 1) Save JSON file
        checkins_dir = Path(__file__).parent.parent / "checkins"
        checkins_dir.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        file_path = checkins_dir / f"checkin-{ts}.json"

        data = {
            "mood": checkin.mood,
            "stress": checkin.stress,
            "sleepHours": checkin.sleepHours,
            "activityMinutes": checkin.activityMinutes,
            "focusArea": checkin.focusArea,
            "reflection": checkin.reflection,
        }

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved wellness check-in JSON to %s", file_path)

        # 2) Summarize back to the user
        summary_prompt = (
            "Summarize this wellness check-in in 2–3 short sentences. "
            "Be encouraging, but do NOT give medical advice or diagnoses.\n\n"
            f"Mood: {checkin.mood}\n"
            f"Stress: {checkin.stress}\n"
            f"Sleep hours: {checkin.sleepHours}\n"
            f"Activity minutes: {checkin.activityMinutes}\n"
            f"Focus area: {checkin.focusArea}\n"
            f"Reflection: {checkin.reflection}\n"
        )

        await self.session.generate_reply(instructions=summary_prompt)

    @function_tool()
    async def set_mood(self, context: RunContext, mood: str):
        """Set the user's current mood in a short word or phrase, e.g. 'tired', 'anxious', 'okay', 'calm'."""
        logger.info("Setting mood to %s", mood)
        self._state["mood"] = mood
        await self._check_completion()

    @function_tool()
    async def set_stress(self, context: RunContext, stress: str):
        """Set the user's stress level, e.g. 'low', 'medium', 'high', or a short phrase."""
        logger.info("Setting stress to %s", stress)
        self._state["stress"] = stress
        await self._check_completion()

    @function_tool()
    async def set_sleep(self, context: RunContext, hours: float):
        """Set how many hours the user slept last night. Use a number like 6.5."""
        logger.info("Setting sleepHours to %s", hours)
        self._state["sleepHours"] = hours
        await self._check_completion()

    @function_tool()
    async def set_activity(self, context: RunContext, minutes: int):
        """Set how many minutes of movement/exercise the user had today."""
        logger.info("Setting activityMinutes to %s", minutes)
        self._state["activityMinutes"] = minutes
        await self._check_completion()

    @function_tool()
    async def set_focus_area(self, context: RunContext, focus: str):
        """Set the user's main focus area, e.g. 'sleep', 'stress', 'fitness', 'diet', 'productivity'."""
        logger.info("Setting focusArea to %s", focus)
        self._state["focusArea"] = focus
        await self._check_completion()

    @function_tool()
    async def set_reflection(self, context: RunContext, reflection: str):
        """Set a short free-text reflection from the user about their day or how they feel."""
        logger.info("Setting reflection")
        self._state["reflection"] = reflection
        await self._check_completion()

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
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
