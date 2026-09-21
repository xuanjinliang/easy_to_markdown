import os
from typing import Any
import re


def ensure_dir(path: str | list[str]):
    if isinstance(path, str):
        path = [path]

    for item in path:
        os.makedirs(item, exist_ok=True)


def chunk_list(data: list[Any], size=10):
    for i in range(0, len(data), size):
        yield data[i:i + size]


def remove_fenced_code_block(text: str) -> str:
    match = re.fullmatch(
        r"\s*```(?:markdown|json|html)?\s*(.*?)\s*```\s*",
        text,
        flags=re.DOTALL,
    )
    return match.group(1) if match else text


def get_area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0

    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])

    return width * height


def calc_overlap_ratio(bbox1: list[float], bbox2: list[float]) -> float:
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (
            (x2 - x1) *
            (y2 - y1)
    )

    area1 = get_area(bbox1)
    area2 = get_area(bbox2)

    if area1 <= 0 or area2 <= 0:
        return 0.0

    return intersection / min(area1, area2)
