from schema import ResearchOutput
from pydantic import ValidationError
import json
from pathlib import Path
from typing import List, Tuple
import base64

# ---------------------------------------------------------------------------
# Pretty Print Helper
# ---------------------------------------------------------------------------
def print_fields(data):
    if isinstance(data, str):
        try:
            data = ResearchOutput(**json.loads(data))
        except (json.JSONDecodeError, ValidationError):
            print("Raw output:", data)
            return

    print("\nObservations:")
    for item in data.observations:
        print(f"- {item}")

    print("\nHypotheses:")
    for item in data.hypotheses:
        print(f"- {item}")

    print("\nSuggested Next Steps:")
    for item in data.next_steps:
        print(f"- {item}")

# ---------------------------------------------------------------------------
# Helpers: Load images as data URLs (for vision inputs)
# ---------------------------------------------------------------------------

def to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image type: {image_path.name}")

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_chapter_images(limit: int = 5) -> list[tuple[str, str]]:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR.parent / "data" / "voynich_chapter_01" 

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found at: {DATA_DIR}\n"
            "Expected: project-root/data/voynich_chapter_01/"
        )

    image_paths = sorted(
        [p for p in DATA_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in: {DATA_DIR}\n"
            "Add files like page_01.png, page_02.png, etc."
        )

    selected = image_paths[:limit]
    return [(p.name, to_data_url(p)) for p in selected]

# ---------------------------------------------------------------------------
# Reasoning loop helpers
# ---------------------------------------------------------------------------

def build_plan_prompt(goal: str) -> str:
    return (
        "Create a short plan of 3 to 5 steps to accomplish the goal below.\n"
        "Each step must be an action the agent can take in this project.\n"
        "Return the plan as a numbered list.\n\n"
        f"Goal: {goal}"
    )

def build_execute_prompt(step: str) -> str:
    return (
        "Execute the step below using available context and any manuscript evidence already provided.\n"
        "Return findings as ResearchOutput JSON.\n\n"
        f"Step: {step}"
    )

def build_evaluate_prompt(goal: str, last_output: ResearchOutput) -> str:
    return (
        "Evaluate progress toward the goal.\n"
        "Answer with:\n"
        "1) Progress: one sentence\n"
        "2) Gaps: one sentence\n"
        "3) Next: either 'continue' or 'stop'\n\n"
        f"Goal: {goal}\n\n"
        f"Latest observations: {last_output.observations}\n"
        f"Latest hypotheses: {last_output.hypotheses}\n"
        f"Latest next_steps: {last_output.next_steps}\n"
    )

def parse_numbered_plan(text: str) -> List[str]:
    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # accepts "1. ..." or "1) ..."
        if len(line) > 2 and (line[0].isdigit() and (line[1] == "." or line[1] == ")")):
            steps.append(line[2:].strip())
    # fallback if model didn't format as numbered list
    if not steps:
        steps = [s.strip("- ").strip() for s in text.splitlines() if s.strip()]
    return [s for s in steps if s]

def should_continue(eval_text: str) -> bool:
    lowered = eval_text.lower()
    # conservative: continue unless the model clearly says stop
    return "next: stop" not in lowered and "next: \"stop\"" not in lowered
