"""
Hands-on AI: Chat with Your Data using ChatGPT
All examples use Python and the OpenAI client.
 
Prereqs:
  pip install -r requirements.txt
  export API_KEY = os.environ[...] or set the api_key to the client
"""

import os
import asyncio

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, ModelSettings, SQLiteSession  # add SQLiteSession
from pathlib import Path

from schema import ResearchOutput
from helpers import ( 
    print_fields, 
    load_chapter_images, 
    pattern_analyzer, 
    translator,
)
# ---------------------------------------------------------------------------
# Create OPENAI Client
# ---------------------------------------------------------------------------

# read local .env file
_ = load_dotenv(find_dotenv()) 

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY not found. Add it to your .env file.")

# retrieve OpenAI API key
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']  
)

# ---------------------------------------------------------------------------
# Load Mission Definition
# ---------------------------------------------------------------------------
MISSION_PATH = Path(__file__).parent / "mission.md"

if not MISSION_PATH.exists():
    raise FileNotFoundError(
        "mission.md not found. Please complete the 'Prepare Your Data and Mission' step first."
    )

mission_text = MISSION_PATH.read_text().strip()

# ---------------------------------------------------------------------------
# Create the Voynich Research Agent
# ---------------------------------------------------------------------------
voynich_agent = Agent(
    name="Voynich Research Agent",
    model="gpt-5.2",
    instructions=(
        f"{mission_text}\n\n"
        "You must follow the mission above at all times.\n"
        "Always return your response as valid JSON using the following structure:\n\n"
        '{'
        '"observations": ["string"], '
        '"hypotheses": ["string"], '
        '"next_steps": ["string"]'
        '}'
    ),
    output_type=ResearchOutput,
    model_settings=ModelSettings(
        reasoning={"effort": "medium"},          # minimal | low | medium | high
        extra_body={"text": {"verbosity": "low"}}  # low | medium | high
    ),
    tools=[pattern_analyzer, translator],
)

# ---------------------------------------------------------------------------
# Interactive Chat Loop
# ---------------------------------------------------------------------------
async def bootstrap_session(session: SQLiteSession) -> None:
    """
    One-time bootstrap: provide the chapter images once so the agent can form baseline observations.
    Those observations are then stored in the SQLiteSession memory.
    """
    chapter_images = load_chapter_images(limit=5)

    initial_content = [
        {
            "type": "input_text",
            "text": (
                "You are about to chat with a chapter of an undeciphered manuscript.\n"
                "First, scan these pages as a single chapter and write baseline findings.\n"
                "Focus on recurring visual structures, layout patterns, and repeated symbol clusters.\n"
                "Do not attempt definitive translation."
            ),
        }
    ]

    for _, data_url in chapter_images:
        initial_content.append({"type": "input_image", "image_url": data_url})

    print("\nBootstrapping agent memory with a visual scan...\n")
    result = await Runner.run(
        voynich_agent,
        [{"role": "user", "content": initial_content}],
        session=session,
    )
    print_fields(result.final_output)


async def chat_loop(session: SQLiteSession) -> None:
    """
    Simple console chat loop.
    The user chats in real time, and the agent uses memory + tools to respond.
    """
    print("\nChat ready. Ask questions about the manuscript chapter.")
    print("Type 'help' for examples, or 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("\nGoodbye.\n")
            return

        if user_text.lower() == "help":
            print(
                "\nTry questions like:\n"
                "What motifs repeat across pages?\n"
                "Do any symbol clusters look like headings?\n"
                "Summarize the strongest hypothesis so far.\n"
                "What should we investigate next and why?\n"
                "Run a quick pattern scan on this snippet: qo aiin chedy qo...\n"
            )
            continue

        # If the user provides a transcription snippet, encourage tool use.
        # Keep it lightweight: no rigid parsing, just a hint.
        hint = ""
        if "qo " in user_text.lower() or "aiin" in user_text.lower() or "chedy" in user_text.lower():
            hint = (
                "\nIf the user is providing a transcription snippet, consider calling "
                "pattern_analyzer and/or translator."
            )

        prompt = (
            f"{user_text}\n"
            f"{hint}\n\n"
            "Return ResearchOutput JSON."
        )

        try:
            result = await Runner.run(
                voynich_agent,
                prompt,
                session=session,
            )
            print_fields(result.final_output)
        except Exception as e:
            print("Error:", e)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main():
    db_path = Path(__file__).parent / "voynich_memory.db"
    session = SQLiteSession("voynich_chapter_01_chat", str(db_path))

    # One-time bootstrap so the chat feels grounded
    await bootstrap_session(session)

    # Start interactive chat
    await chat_loop(session)


if __name__ == "__main__":
    asyncio.run(main())






