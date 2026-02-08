from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Define the Agent Output Schema
# ---------------------------------------------------------------------------
class ResearchOutput(BaseModel):
    observations: list[str]
    hypotheses: list[str]
    next_steps: list[str]