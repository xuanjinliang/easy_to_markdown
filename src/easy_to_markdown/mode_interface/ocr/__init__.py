from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel, Field
from typing import Any

T = TypeVar("T")


class OCRContent(BaseModel):
    input_path: str = ""
    model_settings: dict[str, Any] = Field(default_factory=dict)
    text_det_params: dict[str, Any] = Field(default_factory=dict)
    textline_orientation_angles: list[int] = []
    text_rec_score_thresh: float = 0.0
    content: list[str] = []
    scores: list[float] = []
    bbox: list[list[float]] = []


class OcrInterface(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, x: T):
        pass

    @staticmethod
    @abstractmethod
    def process_item(result: T) -> OCRContent:
        pass

    def advanced_recognition(self, image_list: list[str]) -> list[OCRContent]:
        pass
