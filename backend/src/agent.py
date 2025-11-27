import logging

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional
import re

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

logger = logging.getLogger("sdr_agent")

BASE = Path(__file__).parent.parent  # backend/
SHARED = BASE / "shared data"
FAQ_PATH = SHARED / "lenskart_faq.json"
LEADS_DIR = BASE / "leads"
LEADS_DIR.mkdir(exist_ok=True)

load_dotenv(".env.local")


# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
#             You eagerly assist users with their questions by providing information from your extensive knowledge.
#             Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
#             You are curious, friendly, and have a sense of humor.""",
#         )

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

BASE = Path(__file__).parent.parent
DB = BASE / "shared data" / "airtel_fraud_cases.json"

def load_cases():
    with open(DB, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cases(cases):
    with open(DB, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)


class FraudAgent(Agent):
    def __init__(self, chat_ctx, tts=None):
        base = Path(__file__).parent.parent
        db_path = base / "shared data" / "airtel_fraud_cases.json"

        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Data is a direct list of cases
        self.case = data[0]
        
        self.db_path = db_path

        instructions = (
            "You are a fraud detection representative from Airtel Payments Bank. "
            "Use calm, professional language. Never ask for PINs or full card numbers. "
            "Verify the user only using the security question."
        )

        super().__init__(chat_ctx=chat_ctx, instructions=instructions, tts=tts)

        self.state = "ask_name"
        self.verified = False

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Hello, this is Airtel Payments Bank Fraud Prevention Desk. "
                         "We detected unusual activity on your account. May I know your name?"
        )

    async def handle_turn(self, user_text: str):
        ut = user_text.lower().strip()

        # --- Step 1: Ask for username ---
        if self.state == "ask_name":
            if self.case["userName"].lower() in ut:
                self.state = "verify"
                await self.session.generate_reply(
                    instructions=f"Thank you. Please answer this security question: "
                                 f"{self.case['securityQuestion']}"
                )
            else:
                await self.session.generate_reply(
                    instructions="Sorry, I couldn't match that name. Please state your full name."
                )
            return

        # --- Step 2: Verify ---
        if self.state == "verify":
            if self.case["securityAnswer"].lower() in ut:
                self.state = "ask_confirmation"
                await self.session.generate_reply(
                    instructions=(
                        "Verification successful. Here is the suspicious transaction: "
                        f"A charge of {self.case['transactionAmount']} at {self.case['merchant']} "
                        f"in {self.case['location']} on {self.case['timestamp']} "
                        f"using card ending with {self.case['cardEnding']}. "
                        "Did you make this transaction?"
                    )
                )
            else:
                await self._fail_verification()
            return

        # --- Step 3: Confirm or deny ---
        if self.state == "ask_confirmation":
            if any(x in ut for x in ["yes", "i did", "yeah", "yep"]):
                self._update_case("confirmed_safe", "Customer confirmed the transaction.")
                await self.session.generate_reply(
                    instructions="Thanks for confirming. The transaction is marked as safe. Have a nice day."
                )
            else:
                self._update_case("confirmed_fraud", "Customer denied the transaction.")
                await self.session.generate_reply(
                    instructions="Thank you. We have blocked your card and raised a dispute. "
                                 "Our team will follow up with you shortly."
                )
            return

    async def _fail_verification(self):
        self._update_case("verification_failed", "User failed identity verification.")
        await self.session.generate_reply(
            instructions="Sorry, I cannot verify your identity. "
                         "Please contact Airtel Payments Bank support."
        )

    def _update_case(self, status, note):
        self.case["status"] = status
        self.case["outcomeNote"] = note

        # Save as a list (matching the original structure)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump([self.case], f, indent=2)

        print("Updated fraud case:", self.case)




def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",   # change to en-IN-aish
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    fraud_agent = FraudAgent(chat_ctx=session._chat_ctx, tts=session._tts)

    await session.start(
        agent=fraud_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()



    # # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    # session = AgentSession(
    #     # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
    #     # See all available models at https://docs.livekit.io/agents/models/stt/
    #     stt=deepgram.STT(model="nova-3"),
    #     # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
    #     # See all available models at https://docs.livekit.io/agents/models/llm/
    #     llm=google.LLM(
    #             model="gemini-2.5-flash",
    #         ),
    #     # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
    #     # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
    #     tts=murf.TTS(
    #             voice="en-US-matthew",   # change to en-IN-aishwarya if you want female Indian voice
    #             style="Conversation",
    #             tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    #             text_pacing=True
    #         ),
    #     # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
    #     # See more at https://docs.livekit.io/agents/build/turns
    #     turn_detection=MultilingualModel(),
    #     vad=ctx.proc.userdata["vad"],
    #     # allow the LLM to generate a response while waiting for the end of turn
    #     # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
    #     preemptive_generation=True,
    # )

    # # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # # 1. Install livekit-agents[openai]
    # # 2. Set OPENAI_API_KEY in .env.local
    # # 3. Add `from livekit.plugins import openai` to the top of this file
    # # 4. Use the following session setup instead of the version above
    # # session = AgentSession(
    # #     llm=openai.realtime.RealtimeModel(voice="marin")
    # # )

    # # Metrics collection, to measure pipeline performance
    # # For more information, see https://docs.livekit.io/agents/build/metrics/
    # usage_collector = metrics.UsageCollector()

    # @session.on("metrics_collected")
    # def _on_metrics_collected(ev: MetricsCollectedEvent):
    #     metrics.log_metrics(ev.metrics)
    #     usage_collector.collect(ev.metrics)

    # async def log_usage():
    #     summary = usage_collector.get_summary()
    #     logger.info(f"Usage: {summary}")

    # ctx.add_shutdown_callback(log_usage)

    # # # Add a virtual avatar to the session, if desired
    # # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # # avatar = hedra.AvatarSession(
    # #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # # )
    # # # Start the avatar and wait for it to join
    # # await avatar.start(session, room=ctx.room)

    # # Start the session, which initializes the voice pipeline and warms up the models

    # # Load Lenskart FAQ
    # faq_path = Path(__file__).parent.parent / "shared data" / "lenskart_faq.json"
    # if not faq_path.exists():
    #     logger.warning(f"FAQ file not found at: {faq_path}")

    # # SDR voice (TTS)
    # sdr_tts = murf.TTS(
    #     voice="en-US-matthew",   # change to en-IN-aishwarya if you want female Indian voice
    #     style="Conversation",
    #     tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    #     text_pacing=True
    # )

    # # # IMPORTANT: pass session.chat_ctx (not session)
    # # sdr_agent = SalesAgent(chat_ctx=session.chat_ctx, tts=sdr_tts)

    # # use the underlying chat context object from the session
    # sdr_agent = SalesAgent()


    # # Start session with SDR agent ONLY
    # await session.start(
    #   agent=sdr_agent,
    #   room=ctx.room,
    #   room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    # )




    # # Join the room and connect to the user
    # await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

    print(Path(__file__).parent.parent.resolve())
    print((Path(__file__).parent.parent / "shared data" / "lenskart_faq.json").exists())
