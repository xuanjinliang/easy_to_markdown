from pydantic import BaseModel, Field
from typing import Any, Generic, TypeVar
from abc import ABC, abstractmethod

T = TypeVar("T")

class TablePosition(BaseModel):
    img_index: int = Field(ge=0)
    blocks_index: int = Field(ge=0, default=0)
    crop_path: str = ""
    table_model: list[str] = []
    table_content: list[dict[str, Any]] = []


class TableClassificationInterface(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, x: T):
        pass

    @abstractmethod
    async def predict(self, image_list: list[str]) -> list[str | None]:
        pass