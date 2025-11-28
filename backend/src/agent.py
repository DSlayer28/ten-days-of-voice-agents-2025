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

logger = logging.getLogger("food_ordering_agent")

BASE = Path(__file__).parent.parent  # backend/
SHARED = BASE / "shared data"
MENU_PATH = SHARED / "restaurant_menu.json"
ORDERS_DIR = BASE / "orders"
ORDERS_DIR.mkdir(exist_ok=True)

load_dotenv(".env.local")

# UPDATED BUNDLES: Combining Restaurant Meals + Grocery Recipes
BUNDLES = {
    # Restaurant Bundles (Hot Meals)
    "dinner": ["Butter Chicken", "Butter Roti", "Jeera Rice"],
    "veg meal": ["Paneer Butter Masala", "Naan", "Plain Rice"],
    "party starter": ["Paneer Tikka", "Chicken 65", "Spring Rolls"],
    
    # Grocery Bundles (Ingredients)
    "sandwich": ["Whole Wheat Bread", "Butter", "Cheese Slices"],
    "breakfast": ["Milk (1L)", "Whole Wheat Bread", "Butter"],
    "rice curry": ["Basmati Rice (1kg)", "Tomatoes (1kg)", "Onions (1kg)"]
}
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

class FoodOrderingAgent(Agent):
    def __init__(self):
        # Load menu
        try:
            with open(MENU_PATH, "r", encoding="utf-8") as f:
                self.menu_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Menu file not found at {MENU_PATH}")
            self.menu_data = {"restaurant": {"name": "Default"}, "menu": []}

        self.restaurant_name = self.menu_data["restaurant"].get("name", "Spice Garden & Mart")
        self.categories = self.menu_data.get("menu", [])
        
        instructions = (
            f"You are Alex, a helpful assistant at {self.restaurant_name}. "
            "You can help customers order hot meals from the restaurant OR buy groceries for their home. "
            "If a user asks for 'ingredients', look for grocery items. "
            "If they ask for a 'meal', look for restaurant items. "
            "Be friendly, concise, and helpful. "
            "Start by greeting the customer and asking for their name."
        )
        
        super().__init__(instructions=instructions)
        
        self.customer_name = None
        self.phone_number = None
        self.delivery_address = None
        self.cart = []
        self.order_id = None
        
    async def on_enter(self):
        # 2. INTRO UPDATE: Welcome to the "Mart" as well
        await self.session.generate_reply(
            instructions=(
                f"Hello! Welcome to {self.restaurant_name}. "
                "I'm Alex. I can help you order a hot dinner or just pick up some groceries for the week. "
                "Who am I speaking with?"
            )
        )
    
    async def handle_turn(self, user_text: str):
        ut = user_text.lower().strip()
        
        if self.state == "greeting":
            self.customer_name = user_text.strip()
            self.state = "ordering"
            
            await self.session.generate_reply(
                instructions=(
                    f"Hi {self.customer_name}! "
                    "Are you looking to order a meal to eat now, or do you need to buy some groceries?"
                )
            )
            return
        
        if self.state == "ordering":
            # Checkout
            if any(phrase in ut for phrase in ["that's all", "finish", "checkout", "place order", "done"]):
                if not self.cart:
                    await self.session.generate_reply(
                        instructions="Your cart is empty! Can I get you some fresh veggies or maybe a Biryani?"
                    )
                    return
                
                self.state = "confirm_order"
                cart_summary = self._get_cart_summary()
                await self.session.generate_reply(
                    instructions=(
                        f"Got it. Let's wrap this up. "
                        f"You have: {cart_summary}. "
                        "Does that look right?"
                    )
                )
                return

            # View Cart
            if any(phrase in ut for phrase in ["what's in my cart", "read back", "list items"]):
                cart_summary = self._get_cart_summary()
                await self.session.generate_reply(
                    instructions=f"Currently, you have: {cart_summary}. What else?"
                )
                return

            # 3. BUNDLE LOGIC: Handles "Ingredients for Sandwich" (Grocery) vs "Dinner" (Restaurant)
            added_bundle = self._process_bundle_request(ut)
            if added_bundle:
                items_text = ", ".join(added_bundle)
                await self.session.generate_reply(
                    instructions=(
                        f"Smart choice! I've added all the essentials: {items_text}. "
                        "Anything else?"
                    )
                )
                return

            # Single Item Add
            added_item = self._process_add_request(ut)
            if added_item:
                # 4. SMART SUGGESTIONS: Distinguish Restaurant Upsell vs Grocery Upsell
                upsell = ""
                
                # If they bought a spicy restaurant meal -> Suggest Lassi
                if self._is_spicy_category(added_item):
                    upsell = " A Mango Lassi would cool that down perfectly. Want one?"
                
                # If they bought Grocery Bread -> Suggest Butter (if not in cart)
                elif "bread" in added_item['name'].lower() and not self._has_item_in_cart("butter"):
                    upsell = " Do you need some Butter to go with that bread?"
                
                await self.session.generate_reply(
                    instructions=(
                        f"Added {added_item['name']}.{upsell} "
                        "What's next?"
                    )
                )
            else:
                await self.session.generate_reply(
                    instructions=(
                        "I couldn't find that item. "
                        "We have Restaurant specials and a full Grocery section. "
                        "Could you be more specific?"
                    )
                )
            return
        
        if self.state == "confirm_order":
            if any(word in ut for word in ["yes", "correct", "right", "sure", "yep"]):
                self.state = "get_phone"
                await self.session.generate_reply(
                    instructions="Great. Please share your phone number for the delivery."
                )
            else:
                self.state = "ordering"
                await self.session.generate_reply(
                    instructions="No problem. What item would you like to change?"
                )
            return
        
        if self.state == "get_phone":
            import re
            digits = re.sub(r'\D', '', ut)
            if len(digits) >= 10:
                self.phone_number = digits
                self.state = "get_address"
                await self.session.generate_reply(
                    instructions="Got it. And the delivery address?"
                )
            else:
                await self.session.generate_reply(
                    instructions="Please provide a valid 10-digit phone number."
                )
            return
        
        if self.state == "get_address":
            self.delivery_address = user_text.strip()
            self.state = "finished"
            
            self._save_order()
            total_price = sum(item["price"] * item["quantity"] for item in self.cart)
            
            await self.session.generate_reply(
                instructions=(
                    f"Order #{self.order_id} placed! "
                    f"Total: ₹{total_price}. Delivering to {self.delivery_address}. "
                    "Thank you for shopping at Spice Garden & Mart!"
                )
            )
            return

    # --- Helpers ---

    def _process_add_request(self, text: str) -> Optional[Dict]:
        text = text.lower()
        for category in self.categories:
            for item in category["items"]:
                if item["name"].lower() in text:
                    qty = 1
                    import re
                    match = re.search(r'(\d+)\s+' + re.escape(item["name"].lower()), text)
                    if match:
                        qty = int(match.group(1))

                    cart_item = {
                        "id": item["id"],
                        "name": item["name"],
                        "price": item["price"],
                        "quantity": qty,
                        "category": category["category"]
                    }
                    self.cart.append(cart_item)
                    return cart_item
        return None

    def _process_bundle_request(self, text: str) -> Optional[List[str]]:
        text = text.lower()
        added_names = []
        
        for bundle_name, items in BUNDLES.items():
            if bundle_name in text:
                for item_name in items:
                    menu_item = self._find_item_data(item_name)
                    if menu_item:
                        self.cart.append({
                            "id": menu_item["id"],
                            "name": menu_item["name"],
                            "price": menu_item["price"],
                            "quantity": 1,
                            "category": "Bundle"
                        })
                        added_names.append(menu_item["name"])
                return added_names if added_names else None
        return None

    def _find_item_data(self, name: str) -> Optional[Dict]:
        for category in self.categories:
            for item in category["items"]:
                if item["name"].lower() == name.lower():
                    return item
        return None

    def _is_spicy_category(self, item: Dict) -> bool:
        return item.get("category") in ["Main Course", "Starters"]

    def _has_item_in_cart(self, item_name_part: str) -> bool:
        for item in self.cart:
            if item_name_part.lower() in item["name"].lower():
                return True
        return False

    def _get_cart_summary(self) -> str:
        if not self.cart:
            return "nothing"
        summary = [f"{i['quantity']} {i['name']}" for i in self.cart]
        total = sum(i["price"] * i["quantity"] for i in self.cart)
        return f"{', '.join(summary)} (Total: ₹{total})"

    def _save_order(self):
        self.order_id = f"ORD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        data = {
            "order_id": self.order_id,
            "timestamp": datetime.now().isoformat(),
            "customer": {
                "name": self.customer_name,
                "phone": self.phone_number,
                "address": self.delivery_address
            },
            "items": self.cart,
            "total": sum(i["price"] * i["quantity"] for i in self.cart),
            "status": "placed"
        }
        
        filepath = ORDERS_DIR / f"{self.order_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved order {self.order_id}")



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

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
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
    
    # Initialize agent without passing session internals
    food_agent = FoodOrderingAgent()
    
    await session.start(
        agent=food_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    
    await ctx.connect()
    
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

    print(Path(__file__).parent.parent.resolve())
    print((Path(__file__).parent.parent / "shared data" / "lenskart_faq.json").exists())
