from pydantic import BaseModel, Field
from typing import Any


class TablePosition(BaseModel):
    img_index: int = Field(ge=0)
    blocks_index: int = Field(ge=0, default=0)
    crop_path: str = ""
    table_model: list[str] = []
    table_content: list[dict[str, Any]] = []
