import asyncio
from easy_to_markdown.llm_model import APIModelConfig, TextFormat
from easy_to_markdown.llm_model.interface import LocalModelInterface
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel
from typing import Any, Optional
from easy_to_markdown.llm_model import ModelInfo
from easy_to_markdown.pkg.merge import deep_merge
from easy_to_markdown.pkg.result import Result

import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


class LLMConfig(BaseModel):
    model_id: str
    temperature: int
    max_tokens: int
    stream: bool
    reasoning_effort: Optional[str]
    parallel_tool_calls: bool
    extra_body: dict[str, Any]
    response_format: dict[str, Any] | None = None


class OpenAPIWorker:
    def __init__(self, config: APIModelConfig):
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        self.default_config = config
        self.llm_config = self.set_llm_config(config=config)

    @staticmethod
    def set_llm_config(config: APIModelConfig) -> LLMConfig:
        max_output_tokens = config.max_output_tokens if (
                config.max_output_tokens <= 65535) else 65535

        temperature = 2 if config.temperature > 2 else (
            0 if config.temperature < 0 else config.temperature)

        reasoning_list: list[str] = ["high", "medium", "low", "minimal"]
        reasoning_effort = config.reasoning_effort if (
                config.reasoning_effort is not None and
                config.reasoning_effort in reasoning_list) else None

        llm_config = LLMConfig(
            model_id="Qwen3-VL-4B-Instruct-8bit" if config.model is None else config.model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            stream=config.model_client_stream,
            parallel_tool_calls=config.parallel_tool_calls,
            extra_body={}
        )

        if config.thinking_budget > 0:
            llm_config.extra_body["enable_thinking"] = "true"
            llm_config.extra_body["thinking_budget"] = config.thinking_budget

        if isinstance(config.extra_args, dict):
            llm_config.extra_body = llm_config.extra_body | config.extra_args

        if isinstance(config.text_format, TextFormat):
            llm_config.response_format = config.text_format.model_dump()
        elif isinstance(config.text_format, type) and issubclass(config.text_format, BaseModel):
            openai_schema = to_strict_json_schema(config.text_format)
            llm_config.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "user_profile_schema",
                    "strict": True,
                    "schema": openai_schema
                }
            }
        else:
            raise TypeError("text_format must be TextFormat, BaseModel subclass]")

        return llm_config

    def update_llm_config(self, config: APIModelConfig) -> LLMConfig:
        merged = APIModelConfig.model_validate(
            deep_merge(
                self.default_config.model_dump(),
                config.model_dump(exclude_unset=True)
            )
        )
        return self.set_llm_config(config=merged)

    async def inference(self, messages: list[dict[str, Any]], config: APIModelConfig | None = None) -> Result:

        llm_config = self.llm_config
        if config is not None:
            llm_config = self.update_llm_config(config=config)

        result = ModelInfo(
            model_version=llm_config.model_id
        )

        try:

            params = {
                "model": llm_config.model_id,
                "messages": messages,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
                "stream": llm_config.stream,
                "reasoning_effort": llm_config.reasoning_effort,
                "parallel_tool_calls": llm_config.parallel_tool_calls,
                "extra_body": llm_config.extra_body,
                "response_format": llm_config.response_format
            }

            if llm_config.stream:
                params["stream_options"] = {"include_usage": True}

            response = await self.client.chat.completions.create(**params)

            answer_content = ""
            input_tokens = 0
            output_tokens = 0
            reasoning_content = ""

            if llm_config.stream:
                async for chunk in response:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            answer_content += delta.content

                        if delta.model_extra and delta.model_extra['reasoning_content']:
                            reasoning_content += delta.model_extra['reasoning_content']

                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens

                await response.close()
            else:
                if not response.choices:
                    logger.info(f"Usage:{response.usage}")
                else:
                    answer_content = response.choices[0].message.content

                if response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens

            logger.info(f"reasoning_content:{reasoning_content}")

            result.content = answer_content
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens

            return Result(success=True, result=result, error=None)
        except Exception as e:
            return Result(success=False, result=result, error={e})


class LLMServiceApi(LocalModelInterface):
    def __init__(self, config: APIModelConfig):
        self.client = OpenAPIWorker(config=config)
        self.infer_semaphore = asyncio.Semaphore(config.workers)

    def preprocess_image(self, prompt: str | None = None, images: list[str] | None = None) -> list[dict[str, Any]]:
        content = []

        if images is not None and len(images) > 0:
            for image in images:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": image,
                    }
                )

        if prompt is not None and len(prompt) > 0:
            content.append({"type": "input_text", "text": prompt})

        return self.generate_message(content=content)

    @staticmethod
    def generate_message(content: list[dict[str, Any]] | str, role: str = "user") -> list[dict[str, Any]]:
        return [
            {
                "role": role,
                "content": content
            }
        ]

    async def handle_item(self, message: list[dict[str, Any]], schema: type[BaseModel] | None) -> Result:
        config = None
        if schema is not None:
            config = APIModelConfig(
                text_format=schema
            )

        async with self.infer_semaphore:
            results = await self.client.inference(message, config=config)

        return results

    async def request_vllm(self,
                           messages: list[list[dict[str, Any]]],
                           schema: type[BaseModel] | None = None) -> list[Result]:
        if not messages:
            return []

        tasks = []

        for message in messages:
            task = asyncio.create_task(self.handle_item(message, schema))
            tasks.append(task)

        return await asyncio.gather(*tasks)
