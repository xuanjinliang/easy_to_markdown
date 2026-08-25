from pydantic import BaseModel, Field
from typing import Literal, Type, Union, Any
import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


class TextFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"

class LocalModelConfig(BaseModel):
    model_path: str
    max_output_tokens: int = 8192
    temperature: int = 1
    workers: int = 1


class APIModelConfig(BaseModel):
    model: str | None = None
    base_url: str = ""
    api_key: str = ""
    max_output_tokens: int = 8192
    thinking_budget: int = 5000
    temperature: int = 1
    reasoning_effort: str = "medium"
    max_retry: int = 3
    model_client_stream: bool = True
    parallel_tool_calls: bool = True
    text_format: Union[
        TextFormat,
        Type[BaseModel]
    ] = TextFormat()
    extra_args: dict[str, Any] = Field(default_factory=dict)
    workers: int = 4


class ModelInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    model_version: str = ""
    content: Any = None


class AdvancedOCRVL(BaseModel):
    location: list[float] = []
    rotate_rect: list[float] = []
    text: str = ""


class ClientConfig(BaseModel):
    model: str | None = None
    max_output_tokens: int = 8192
    thinking_budget: int = 5000
    temperature: int = 1
    reasoning_effort: str = "medium"
    max_retry: int = 3
    model_client_stream: bool = True
    parallel_tool_calls: bool = True
    text_format: Union[
        TextFormat,
        Type[BaseModel]
    ] = TextFormat()
    extra_args: dict[str, Any] = Field(default_factory=dict)


class ModelAdapter:
    @staticmethod
    def common_config(config: ClientConfig) -> ClientConfig:
        temperature = config.temperature
        max_output_tokens = config.max_output_tokens
        thinking_budget = config.thinking_budget

        if max_output_tokens > 65535:
            max_output_tokens = 65535
        if max_output_tokens <= 0:
            max_output_tokens = 8192

        if temperature > 2:
            temperature = 2
        elif temperature < 0:
            temperature = 0

        if thinking_budget > 15000:
            thinking_budget = 15000
        elif thinking_budget < 0:
            thinking_budget = -1
        elif thinking_budget < 1024:
            thinking_budget = 0

        config.temperature = temperature
        config.max_output_tokens = max_output_tokens
        config.thinking_budget = thinking_budget
        return config
