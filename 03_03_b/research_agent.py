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
from helpers import print_fields, load_chapter_images, build_plan_prompt, parse_numbered_plan, build_execute_prompt, build_evaluate_prompt, should_continue

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
        # Create a persistent SQLite session (writes to a local db file)
        db_path = Path(__file__).parent / "voynich_memory.db"
        session = SQLiteSession("voynich_chapter_01", str(db_path))

        # Load up to 5 images (your helper should resolve the correct data directory)
        chapter_images = load_chapter_images(limit=5)

        # Provide images only once; memory carries forward what was found
        initial_content = [
            {
                "type": "input_text",
                "text": (
                    "Review these manuscript pages as a single chapter. "
                    "Identify recurring symbols, layout patterns, and structural similarities. "
                    "Do not attempt translation."
                ),
            }
        ]
        for _, data_url in chapter_images:
            initial_content.append({"type": "input_image", "image_url": data_url})

        print("\n--- Step 0: Establish initial findings (with images) ---\n")
        baseline = await Runner.run(
            voynich_agent,
            [{"role": "user", "content": initial_content}],
            session=session
        )
        print_fields(baseline.final_output)

        # -------------------------------------------------------------------
        # Agentic reasoning loop: Plan -> Execute -> Evaluate -> Repeat
        # -------------------------------------------------------------------

        # TODO: Write a clear research goal for your agent
        # goal = ""
        max_steps = 4

        print("\n--- Step 1: Agent creates a plan ---\n")
        plan_result = await Runner.run(
            voynich_agent,
            #TODO: Ask the agent to generate a short plan for achieving the goal
            # using the build_plan_prompt function.
            session=session
        )
        plan_text = plan_result.final_output if isinstance(plan_result.final_output, str) else str(plan_result.final_output)
        print(plan_text)

        plan_steps = parse_numbered_plan(plan_text)
        plan_steps = plan_steps[:max_steps]

        for i, step in enumerate(plan_steps, start=1):
            print(f"\n--- Step 2.{i}: Execute plan step ---\n")
            exec_result = await Runner.run(
                voynich_agent,
                build_execute_prompt(step),
                session=session
            )

            if isinstance(exec_result.final_output, ResearchOutput):
                print_fields(exec_result.final_output)
                last_output = exec_result.final_output
            else:
                # If coercion fails, still print what we got
                print(exec_result.final_output)
                break

            print(f"\n--- Step 3.{i}: Evaluate progress ---\n")
            eval_result = await Runner.run(
                voynich_agent,
                # TODO: Tell the agent to evaluate it's progress toward the goal
                # using the build_evaluate_prompt function
                session=session
            )
            eval_text = eval_result.final_output if isinstance(eval_result.final_output, str) else str(eval_result.final_output)
            print(eval_text)

            #TODO Determine if the loop should terminate using the 
            # should_continue function
    except Exception as e:
        print("Error", e)

if __name__ == "__main__":
    asyncio.run(main())