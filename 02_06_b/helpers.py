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
    #TODO: Implement this function to load the images found in the data folder