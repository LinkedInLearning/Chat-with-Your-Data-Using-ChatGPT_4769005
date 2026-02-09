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
from agents import Agent, Runner, ModelSettings
from pathlib import Path

from schema import ResearchOutput

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
# TODO: Load your mission.md file and store in MISSION_PATH 


mission_text = MISSION_PATH.read_text().strip()

# ---------------------------------------------------------------------------
# Create the Voynich Research Agent
# ---------------------------------------------------------------------------
voynich_agent = Agent(
    # TODO: Give your base agent a name
    #name="",
    model=#TODO: Select the model,
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
    # TODO: Attach the ResearchOutput schema to your agent
    # output_type=
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
        
        # Run through the Agents SDK
        result = await Runner.run(
            voynich_agent,
            [{"role": "user", "content": "Confirm that you're ready to assist us."}]
        )

        print(result.final_output)
    except Exception as e:
        print("Error", e)

if __name__ == "__main__":
    asyncio.run(main())