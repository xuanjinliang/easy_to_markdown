from pydantic import BaseModel, Field
from typing import Any


class Result(BaseModel):
    success: bool
    result: Any = Field(default=None)
    error: Any = Field(default=None)

class AgentResult(BaseModel):
    thought_process: str
    final_result: str