import logging

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from livekit.agents import Agent, AgentTask, function_tool, RunContext
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

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

# ...

@dataclass
class PatientCheckin:
    name: str
    age: int
    mood: str
    stress: str
    sleepHours: float
    activityMinutes: int
    mainConcern: str
    reflection: str



class WellnessCheckinTask(AgentTask[PatientCheckin]):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            instructions=(
                "You are conducting a simple wellness check-in.\n"
                "You MUST use tools to record: name, age, mood, stress, sleepHours, "
                "activityMinutes, mainConcern, reflection.\n"
                "Ask one short question at a time. Keep a supportive but non-clinical tone.\n"
                "Do NOT give diagnoses or medical treatment advice. For serious issues, "
                "tell the user to contact a doctor or local emergency services.\n"
            ),
            **kwargs,
        )
        self._state: Dict[str, Any] = {}

    async def _check_completion(self) -> None:
        required = {
            "name",
            "age",
            "mood",
            "stress",
            "sleepHours",
            "activityMinutes",
            "mainConcern",
            "reflection",
        }

        if required.issubset(self._state.keys()):
            result = PatientCheckin(
                name=str(self._state["name"]),
                age=int(self._state["age"]),
                mood=str(self._state["mood"]),
                stress=str(self._state["stress"]),
                sleepHours=float(self._state["sleepHours"]),
                activityMinutes=int(self._state["activityMinutes"]),
                mainConcern=str(self._state["mainConcern"]),
                reflection=str(self._state["reflection"]),
            )
            # This completes the task and returns control to the agent
            self.complete(result)
        else:
            await self.session.generate_reply(
                instructions="Continue the check-in and ask about any missing information."
            )

    @function_tool()
    async def set_name(self, context: RunContext, name: str):
        """Set the patient's name."""
        self._state["name"] = name
        await self._check_completion()

    @function_tool()
    async def set_age(self, context: RunContext, age: int):
        """Set the patient's age in years."""
        self._state["age"] = age
        await self._check_completion()

    @function_tool()
    async def set_mood(self, context: RunContext, mood: str):
        """Set the current mood in a short word/phrase (e.g. 'okay', 'stressed', 'calm')."""
        self._state["mood"] = mood
        await self._check_completion()

    @function_tool()
    async def set_stress(self, context: RunContext, stress: str):
        """Set stress level: low, medium, high, or a short description."""
        self._state["stress"] = stress
        await self._check_completion()

    @function_tool()
    async def set_sleep_hours(self, context: RunContext, hours: float):
        """Set how many hours the patient slept last night."""
        self._state["sleepHours"] = hours
        await self._check_completion()

    @function_tool()
    async def set_activity_minutes(self, context: RunContext, minutes: int):
        """Set how many minutes of movement/exercise they had today."""
        self._state["activityMinutes"] = minutes
        await self._check_completion()

    @function_tool()
    async def set_main_concern(self, context: RunContext, concern: str):
        """Set the patient's main wellness concern (e.g. sleep, focus, stress)."""
        self._state["mainConcern"] = concern
        await self._check_completion()

    @function_tool()
    async def set_reflection(self, context: RunContext, reflection: str):
        """Set a short free-text reflection from the patient."""
        self._state["reflection"] = reflection
        await self._check_completion()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a calm, supportive wellness companion.\n"
                "You will run a wellness check-in task to collect patient details.\n"
                "Keep your tone practical and supportive, not overly emotional.\n"
            )
        )

    async def on_enter(self) -> None:
        # Greet and explain what will happen
        await self.session.generate_reply(
            instructions=(
                "Greet the user briefly and explain that you'll ask a few questions "
                "to understand how they are doing today. Then start the check-in."
            )
        )

        # Run the task – this will block until all details are collected
        checkin: PatientCheckin = await WellnessCheckinTask(chat_ctx=self.chat_ctx)

        # Save to JSON
        file_path = self._save_checkin_to_file(checkin)

        # Speak a short summary back to the user
        summary = (
            f"Thanks, {checkin.name}. Today you're feeling {checkin.mood} with stress level {checkin.stress}. "
            f"You slept about {checkin.sleepHours} hours and moved for roughly {checkin.activityMinutes} minutes. "
            f"Your main concern right now is {checkin.mainConcern}."
        )

        await self.session.generate_reply(
            instructions=(
                "Summarize the check-in to the user using this summary text in 2–3 short sentences. "
                "Do not give medical advice.\n"
                f"{summary}\n"
                f"Optionally, mention that their check-in has been saved."
            )
        )

    def _save_checkin_to_file(self, checkin: PatientCheckin) -> Path:
        checkins_dir = Path(__file__).parent.parent / "checkins"
        checkins_dir.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        file_path = checkins_dir / f"checkin-{ts}.json"

        data = {
            "name": checkin.name,
            "age": checkin.age,
            "mood": checkin.mood,
            "stress": checkin.stress,
            "sleepHours": checkin.sleepHours,
            "activityMinutes": checkin.activityMinutes,
            "mainConcern": checkin.mainConcern,
            "reflection": checkin.reflection,
        }

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved wellness check-in JSON to %s", file_path)
        return file_path


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
