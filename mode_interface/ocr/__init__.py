from pydantic import BaseModel, Field
from typing import Optional, Literal, Any


class OCRContent(BaseModel):
    input_path: str = ""
    model_settings: dict[str, Any] = Field(default_factory=dict)
    text_det_params: dict[str, Any] = Field(default_factory=dict)
    textline_orientation_angles: list[int] = []
    text_rec_score_thresh: float = 0.0
    content: list[str] = []
    scores: list[float] = []
    bbox: list[list[float]] = []
