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

from pydantic import ValidationError
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
# Run the Agent
# ---------------------------------------------------------------------------
async def main():
    try:
        db_path = Path(__file__).parent / "voynich_memory.db"
        session = SQLiteSession("voynich_chapter_01", str(db_path))

        # Load images once (used for initial visual inspection)
        chapter_images = load_chapter_images(limit=5)

        initial_content = [
            {
                "type": "input_text",
                "text": (
                    "Skim these manuscript pages and identify 2 to 3 recurring visual structures "
                    "that could be worth transcribing (for example, repeated word-like clusters, "
                    "consistent line spacing, recurring marginal marks). Do not attempt translation."
                ),
            }
        ]
        for _, data_url in chapter_images:
            initial_content.append({"type": "input_image", "image_url": data_url})

        print("\n--- Pass 1: Visual scan (with images) ---\n")
        scan_result = await Runner.run(
            voynich_agent,
            [{"role": "user", "content": initial_content}],
            session=session
        )
        print_fields(scan_result.final_output)

        # Simulated transcription snippet (replace later with OCR/vision-to-text)
        simulated_transcription = "qo aiin chedy qo daiin chedy ol qo chedy aiin qo chedy chedy ol"

        tool_prompt = (
            "I’m going to give you a simulated transcription snippet from one region of the chapter.\n\n"
            f"Transcription:\n{simulated_transcription}\n\n"
            "Do the following:\n"
            "1) Call pattern_analyzer on the transcription and summarize the strongest repeated token and bigram patterns.\n"
            "2) Call translator on the same transcription using the default mapping.\n"
            "3) Based on both tool outputs, propose 2 hypotheses and 2 next steps.\n"
            "Return ResearchOutput JSON."
        )

        print("\n--- Pass 2: Tool-assisted analysis (pattern + translation) ---\n")
        tool_result = await Runner.run(voynich_agent, tool_prompt, session=session)
        print_fields(tool_result.final_output)
    except Exception as e:
        print("Error", e)

if __name__ == "__main__":
    asyncio.run(main())