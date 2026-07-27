from abc import ABC, abstractmethod
from typing import Any, Optional, Generic, TypeVar
from llm_model import ClientConfig
from pkg.result import Result
import uuid

T = TypeVar("T")


class LLMInterface(ABC, Generic[T]):
    @abstractmethod
    def update_llm_config(self, config: ClientConfig):
        pass

    @abstractmethod
    def get_batch_file_path(self):
        pass

    @abstractmethod
    def open_file(self, filename: str):
        pass

    @abstractmethod
    def write_json_to_file(self, json_data: dict[str, Any], filename: str = str(uuid.uuid4())):
        pass

    @abstractmethod
    def generate_image(self, prompt: str, images: list[str] | None = None) -> list[T]:
        pass

    @abstractmethod
    def generate_text(self, prompt_list: list[str]) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def request_llm(self,
                          system: Optional[str] = None,
                          messages: list[dict[str, Any]] | None = None,
                          seed: Optional[int] = None,
                          config: ClientConfig | None = None
                          ) -> Result:
        pass
