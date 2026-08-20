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
