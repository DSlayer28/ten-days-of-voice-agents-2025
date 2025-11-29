import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("game_master_agent")

BASE = Path(__file__).parent.parent  # backend/
SHARED = BASE / "shared data"
ORDERS_DIR = BASE / "orders"
ORDERS_DIR.mkdir(exist_ok=True)

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

# ---------------- Day 8 Game Master Agent ----------------
class GameMasterAgent(Agent):
    def __init__(self):
        # Load world model if present
        try:
            path = Path(__file__).parent.parent / "shared data" / "day8_adventure.json"
            with path.open("r", encoding="utf-8") as fh:
                self.world = json.load(fh)
        except Exception as e:
            logger.warning(f"Could not load world model: {e}")
            self.world = None

        instructions = (
            "You are a Game Master running a dark, atmospheric jungle adventure called "
            "'Jungle of the Fallen Gods'. Keep tone eerie, dangerous yet exciting. "
            "After every descriptive line, end with a direct prompt: 'What do you do?' "
            "Keep responses short (1-3 sentences). Use the conversation history to maintain continuity, "
            "remember named choices, locations, and inventory. "
            "Accept direct player commands (look, go, take, attack, hide, inventory, restart)."
        )
        super().__init__(instructions=instructions)

        # Game state
        self.player_name: Optional[str] = None
        self.location: str = None
        self.inventory: List[str] = []
        self.flags: Dict[str, bool] = {}
        self.turn_count = 0
        self.started = False

    async def on_enter(self):
        """Called automatically by LiveKit when agent enters the room"""
        logger.info("GameMasterAgent has entered the room.")
        try:
            # Set up initial game state
            if self.world:
                intro = self.world.get("intro", "")
                self.location = self.world.get("starting_location")
            else:
                intro = (
                    "You step into a dense forest where shadows seem to move with a mind of their own. "
                    "The canopy swallows the light. Something watches from the trees."
                )
                self.location = "edge"

            self.started = True
            self.turn_count = 0
            
            # Use self.say() to speak to the user
            await self.session.generate_reply(instructions=f"{intro}\n\nWhat do you do?")

        except Exception as e:
            logger.error(f"Error in on_enter: {e}")
            await self.session.generate_reply(instructions="Welcome to the jungle. What do you do?")

    # ---------- Helper Methods ----------
    def _location_description(self) -> str:
        """Get description of current location"""
        if not self.world:
            return "The jungle is dense and the air smells of wet earth. Faint drums beat far away."
        
        loc = self.world.get("locations", {}).get(self.location, {})
        return loc.get("desc", "You see thick jungle around you.")

    def _move_to(self, target: str) -> bool:
        """Move player to a new location"""
        if not self.world:
            # Simple fallback movement
            if "path" in target or "clearing" in target or "ruin" in target:
                self.location = "damp_clearing" if "clearing" in target or "path" in target else "ancient_ruin"
                self.flags["beast_alert"] = (self.location == "ancient_ruin")
                return True
            return False

        # Check world model for valid moves
        for name, loc in self.world.get("locations", {}).items():
            if target in name or target in loc.get("desc", "").lower() or target == name:
                # Verify adjacency
                exits = loc.get("exits", [])
                current_exits = self.world.get("locations", {}).get(self.location, {}).get("exits", [])
                
                if self.location in exits or name in current_exits:
                    self.location = name
                    self.flags["beast_alert"] = (name == "ancient_ruin")
                    return True
        return False

    def _take_item(self, obj: str) -> bool:
        """Take an item from current location"""
        if self.world and self.location == "ancient_ruin" and ("relic" in obj or "fragment" in obj or "idol" in obj):
            if "relic fragment" not in self.inventory:
                self.inventory.append("relic fragment")
                self.flags["obtained_relic"] = True
                return True
        
        # Fallback: allow blade in clearing
        if not self.world and self.location == "damp_clearing" and "blade" in obj:
            if "rusty blade" not in self.inventory:
                self.inventory.append("rusty blade")
                return True
        
        return False

    def save_session_log(self, history: List[Dict[str, str]]):
        """Save session log for replay"""
        try:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            fpath = Path(__file__).parent.parent / "orders" / f"gm_session_{ts}.json"
            with open(fpath, "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)
            logger.info(f"Saved GM session log: {fpath}")
        except Exception as e:
            logger.exception(f"Failed to save GM session log: {e}")


# ---------------- LiveKit Entry Point ----------------
# ---------------- end GameMasterAgent ----------------




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

def prewarm(proc: JobProcess):
    """Prewarm models before starting"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent"""
    try:
        # Initialize the session
        session = AgentSession(
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(
                voice="en-US-matthew",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata["vad"],
            preemptive_generation=True,
        )

        # Create game master agent
        gm = GameMasterAgent()

        # Start session - LiveKit will automatically call on_enter()
        await session.start(
            agent=gm,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC()
            ),
        )
        # Connect to room
        await ctx.connect()
        
        logger.info("Game Master agent started successfully")
        
    except Exception as e:
        logger.error(f"Error in entrypoint: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))