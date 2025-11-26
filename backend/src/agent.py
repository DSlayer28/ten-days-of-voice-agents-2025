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

def load_faq() -> Dict[str, Any]:
    if not FAQ_PATH.exists():
        logger.warning("FAQ file missing: %s", FAQ_PATH)
        return {"company": {"name": "Lenskart"}, "faq_entries": []}
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def find_faq_answer_simple(faq_entries, text: str) -> Optional[str]:
    txt = text.lower()
    best_score = 0
    best_answer = None
    for e in faq_entries:
        hay = (e.get("question","") + " " + e.get("answer","")).lower()
        tokens = set(re.findall(r"\w+", hay))
        score = sum(1 for t in tokens if t in txt)
        if score > best_score:
            best_score = score
            best_answer = e.get("answer")
    if best_score > 0:
        return best_answer
    return None

def save_lead(lead: Dict[str,Any]) -> str:
    lead_copy = dict(lead)
    lead_copy["created_at"] = datetime.utcnow().isoformat() + "Z"
    # sanitize name for filename
    name = lead_copy.get("name", "unknown").replace(" ", "_")
    fname = LEADS_DIR / f"lead_{name}_{lead_copy['created_at'].replace(':','-')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(lead_copy, f, indent=2)
    logger.info("Saved lead to %s", fname)
    return str(fname)

class SalesAgent(Agent):
    
    def __init__(self):
        kb = load_faq()
        instructions = (
            "You are Lenskart SDR. Use ONLY the provided FAQ content to answer product/company/pricing questions. "
            "Collect lead fields naturally: name, company, email, role, use_case, team_size, timeline. "
            "Once you have collected information, use the save_lead_info tool to save it. "
            "Confirm key fields before saving. If user asks for info not in FAQ, say: 'I don't have that detail here — would you like me to check the site or get a sales rep to follow up?'"
        )
        super().__init__(instructions=instructions)
        self.kb = kb
        self.faq_entries = kb.get("faq_entries", [])
        self.company = kb.get("company", {"name":"Lenskart"})
        self._lead: Dict[str,Any] = {}

    async def on_enter(self):
        await self.session.generate_reply(instructions=(
            f"Hi — welcome to {self.company.get('name','Lenskart')}! I'm the SDR here. What brought you here today and what are you trying to achieve?"
        ))
    
    @function_tool
    async def save_lead_info(
        self,
        context: RunContext,
        name: Optional[str] = None,
        company: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        use_case: Optional[str] = None,
        team_size: Optional[str] = None,
        timeline: Optional[str] = None
    ):
        """Save lead information collected during the conversation.
        
        Args:
            name: Full name of the lead
            company: Company name
            email: Email address
            role: Job role/title
            use_case: What they want to use Lenskart for
            team_size: Size of their team
            timeline: When they plan to start/purchase
        """
        if name:
            self._lead["name"] = name
        if company:
            self._lead["company"] = company
        if email:
            self._lead["email"] = email
        if role:
            self._lead["role"] = role
        if use_case:
            self._lead["use_case"] = use_case
        if team_size:
            self._lead["team_size"] = team_size
        if timeline:
            self._lead["timeline"] = timeline
        
        # Save to file
        filepath = save_lead(self._lead)
        logger.info(f"Lead saved to {filepath}")
        
        return f"Lead information saved successfully for {self._lead.get('name', 'contact')}"

    async def handle_turn(self, user_text: str):
        ut = user_text.strip()
        # End-of-call detection
        if re.search(r"\b(thats all|that's all|i'm done|im done|thanks|thank you|bye)\b", ut.lower()):
            # confirm unsaved required fields? just produce summary and save whatever collected
            summary = (
                f"Summary: {self._lead.get('name','Unknown')} from {self._lead.get('company','Unknown')}. "
                f"Role: {self._lead.get('role','Unknown')}. Use case: {self._lead.get('use_case','Not specified')}. "
                f"Timeline: {self._lead.get('timeline','Not specified')}."
            )
            save_lead(self._lead)
            await self.session.generate_reply(instructions=f"{summary} I saved this lead and will pass it to our sales team. Thanks!")
            return

        # Try FAQ answer first
        answer = find_faq_answer_simple(self.faq_entries, ut)
        if answer:
            await self.session.generate_reply(instructions=answer)
            # continue to collect lead info if user provides it in same utterance

        # Slot-filling: email
        email_match = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", ut)
        if email_match:
            self._lead["email"] = email_match.group(1)
            await self.session.generate_reply(instructions=f"Got your email as {self._lead['email']}, correct?")
            return

        # Slot-filling: name
        name_match = re.search(r"(my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", ut)
        if name_match:
            self._lead["name"] = name_match.group(2)
            await self.session.generate_reply(instructions=f"Nice to meet you, {self._lead['name']}. Which company are you with?")
            return

        # If user answers a confirmation like "yes" to previous question, store if applicable (simple)
        if ut.lower() in {"yes", "yep", "correct", "that's right", "right"} and "email" in self._lead and not self._lead.get("_email_confirmed"):
            self._lead["_email_confirmed"] = True
            await self.session.generate_reply(instructions="Thanks — email confirmed. What's your role at the company?")
            return

        # Ask for next missing field
        for field, prompt in [
            ("name", "Could I get your full name, please?"),
            ("company", "Which company are you with?"),
            ("email", "What is the best email to reach you on?"),
            ("role", "What's your role there?"),
            ("use_case", "Briefly, what do you want to use Lenskart for?"),
            ("team_size", "How big is your team? (number or small/medium/large)"),
            ("timeline", "When are you planning to start or purchase? (now / soon / later)")
        ]:
            if field not in self._lead:
                await self.session.generate_reply(instructions=prompt)
                return

        # All collected
        await self.session.generate_reply(instructions="Thanks — I have your details. Anything else I can answer about Lenskart?")

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
                voice="en-US-matthew",   # change to en-IN-aishwarya if you want female Indian voice
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

    # Load Lenskart FAQ
    faq_path = Path(__file__).parent.parent / "shared data" / "lenskart_faq.json"
    if not faq_path.exists():
        logger.warning(f"FAQ file not found at: {faq_path}")

    # SDR voice (TTS)
    sdr_tts = murf.TTS(
        voice="en-US-matthew",   # change to en-IN-aishwarya if you want female Indian voice
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True
    )

    # # IMPORTANT: pass session.chat_ctx (not session)
    # sdr_agent = SalesAgent(chat_ctx=session.chat_ctx, tts=sdr_tts)

    # use the underlying chat context object from the session
    sdr_agent = SalesAgent()


    # Start session with SDR agent ONLY
    await session.start(
      agent=sdr_agent,
      room=ctx.room,
      room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )




    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

    print(Path(__file__).parent.parent.resolve())
    print((Path(__file__).parent.parent / "shared data" / "lenskart_faq.json").exists())
