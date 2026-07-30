from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")


class LayoutResult(BaseModel):
    block_id: int = Field(ge=0)
    block_label: str
    block_bbox: list[float] = Field(..., min_length=4, max_length=4)
    block_bbox_norm: list[float] = Field(..., min_length=4, max_length=4)
    reading_order: Optional[int] = None
    group_id: int = Field(ge=0)
    score: float


class LayoutModel(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, x: T):
        pass

    @staticmethod
    @abstractmethod
    def process_item(result: T) -> list[LayoutResult]:
        pass

    def inference(self, image_list: list[str]) -> list[list[LayoutResult]]:
        pass
