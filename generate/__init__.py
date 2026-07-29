from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Literal
from pkg.pdf_to_image import ImageResponse
from mode_interface.ocr import OCRContent


class ParsingResult(BaseModel):
    block_id: int = Field(ge=0)
    block_label: str
    block_label_type: str = ""
    level: int = 0
    block_content: str = Field(default="")
    block_bbox: list[float] = Field(..., min_length=4, max_length=4)
    block_bbox_norm: list[float] = Field(..., min_length=4, max_length=4)
    reading_order: Optional[int] = None
    group_id: int = Field(ge=0)
    score: float
    remove: bool = False
    remove_reason: str = ""
    crop_path: Optional[str] = Field(default=None)
    crop_bbox: Optional[list[float]] = Field(
        default=None,
        min_length=4,
        max_length=4
    )
    ocr_content: list[OCRContent] = []
    table_model: list[str] = []
    table_info: Optional[TableInfo] = None


class CellInfo(BaseModel):
    uid: int = 0
    cls_id: int = 0
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    label: str = ""
    score: float = 0
    width: float = 0
    height: float = 0
    cx: float = 0
    cy: float = 0
    area: float = 0


class ContainerInfo(BaseModel):
    width: float = 0
    height: float = 0
    bbox: list[float] = Field(..., min_length=4, max_length=4)


class ColumnsInfo(ContainerInfo):
    image_path: str
    columns_list: list[CellInfo]
    category_type: Literal["formula", "text", "unknown"] = "text"
    ocr_content: list[OCRContent] = []
    columns_blocks: FileParsingResult | None = None


class RowInfo(ContainerInfo):
    rows_list: list[ColumnsInfo] = []


class TableInfo(ContainerInfo):
    draw_img_path: str
    image_path: str  # Image path
    img_output_dir: str  # Image dir
    table_list: list[RowInfo] = []


class FileParsingResult(BaseModel):
    img_info: ImageResponse
    vis_path: Optional[str] = ""
    blocks: list[ParsingResult]


class TableCellCategory(BaseModel):
    category_type: Literal["formula", "text", "unknown"]
    parsing_result: FileParsingResult | None = None
