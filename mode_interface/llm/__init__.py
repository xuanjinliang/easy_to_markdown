from pydantic import BaseModel
from llm_model import ModelInfo


class LLMContent(BaseModel):
    input_path: str = ""
    content: ModelInfo | None = None
