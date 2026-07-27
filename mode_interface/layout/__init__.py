from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from generate import ParsingResult, ImageResponse

T = TypeVar("T")


class LayoutModel(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, x: T):
        pass

    @staticmethod
    @abstractmethod
    def process_item(result: T) -> list[ParsingResult]:
        pass

    def inference(self, image_list: list[ImageResponse]) -> list[list[ParsingResult]]:
        pass
