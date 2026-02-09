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
from helpers import print_fields, load_chapter_images

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
    )
)

# ---------------------------------------------------------------------------
# Run the Agent
# ---------------------------------------------------------------------------
async def main():
    try:
        # TODO: Create a persistent SQLite session database called voynich_memory.db

        # Load up to 5 images (your helper should resolve the correct data directory)
        chapter_images = load_chapter_images(limit=5)

        # User message contains ONLY the task + the data (images).
        # First prompt: establish initial findings
        # TODO: Write the first_prompt for your agent

        # Second prompt: build on memory without re-supplying images
        # TODO: Write the second_prompt for your agent

        for _, data_url in chapter_images:
            first_prompt.append({"type": "input_image", "image_url": data_url})

        # Run frist prompt through the Agents SDK
        result = await Runner.run(
            voynich_agent,
            [{"role": "user", "content": first_prompt}],
            session=session
        )

        # final_output should be a ResearchOutput instance.
        print_fields(result.final_output)

        print("\n--- Second pass (uses memory) ---\n")
        # Run second prompt through the Agents SDK
        result_2 = await Runner.run(
            voynich_agent,
            [{"role": "user", "content": second_prompt}],
            session=session
        )

        # final_output should be a ResearchOutput instance.
        print_fields(result_2.final_output)
    except Exception as e:
        print("Error", e)

if __name__ == "__main__":
    asyncio.run(main())