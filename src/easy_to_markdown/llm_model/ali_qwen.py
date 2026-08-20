import base64
import os
from easy_to_markdown import pkg
import dashscope
import json, uuid
from pydantic import BaseModel
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pathlib import Path
from easy_to_markdown.llm_model import logger, ClientConfig, ModelAdapter, TextFormat
from easy_to_markdown.llm_model.interface import LLMInterface
from typing import Any, Literal, Optional
from easy_to_markdown.pkg.files_handle import get_file
from src.easy_to_markdown.pkg.image_handle import get_image_extension
from easy_to_markdown.pkg.result import Result
from easy_to_markdown.pkg.merge import deep_merge

QwenOcrTaskType = Literal[
    "multi_lan",
    "document_parsing",
    "advanced_recognition",
    "text_recognition",
    "table_parsing",
    "formula_recognition"
]


class QwenOcrDashscope:
    def __init__(self, config: ClientConfig):
        self.model_id = "qwen3.5-ocr" if config.model is None else config.model

    @classmethod
    def preprocess_image(cls, image_bytes: bytes, enable_rotate: bool) -> dict[str, Any]:
        image_type = get_image_extension(image_bytes)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "image": f"data:image/{image_type};base64,{image_base64}",
            "enable_rotate": enable_rotate
        }

    @classmethod
    def generate_message(cls, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": content
            }
        ]

    def generate_base64(self,
                        enable_rotate: bool = False,
                        images: list[str] | None = None
                        ) -> list[dict[str, Any]]:
        content = []
        if images is not None and len(images) > 0:
            for image in images:
                images_bytes = get_file(image)
                image_obj = self.preprocess_image(images_bytes, enable_rotate)
                content.append(image_obj)

        return self.generate_message(content=content)

    async def request_llm(
            self, messages:
            list[dict[str, Any]],
            task: QwenOcrTaskType = "text_recognition"
    ) -> Result:

        result = {
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "model_version": self.model_id,
        }

        try:
            dashscope.base_http_api_url = "https://llm-been88thlvgn02hr.cn-beijing.maas.aliyuncs.com/api/v1"
            # dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
            response = await dashscope.AioMultiModalConversation.call(
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                model=self.model_id,
                messages=messages,
                ocr_options={"task": task}
            )

            answer_content = ""
            input_tokens = 0
            output_tokens = 0

            logger.info(f"request_id --> {response.request_id}")
            if response.usage:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

            if response.output.choices:
                if task == "advanced_recognition":
                    answer_content = response.output.choices[0].message.content[0]["ocr_result"]["words_info"]
                else:
                    answer_content = response.output.choices[0].message.content[0]['text']
                # answer_content = response.output.choices[0].message.content[0]["ocr_result"]["words_info"]
                # logger.info(f"text --> {response.output.choices[0].message.content[0]['text']}")

            # if isinstance(answer_content, dict) or isinstance(answer_content, list):
            #     answer_content = json.dump(answer_content, ensure_ascii=False)

            result["text"] = answer_content
            result["input_tokens"] = input_tokens
            result["output_tokens"] = output_tokens
            return Result(success=True, result=result, error=None)
        except Exception as e:
            return Result(success=False, result=result, error={e})


class LLMConfig(BaseModel):
    model_id: str
    temperature: int
    max_tokens: int
    stream: bool
    parallel_tool_calls: bool
    extra_body: dict[str, Any]
    response_format: dict[str, Any] = None


class QwenClient(LLMInterface):
    def __init__(self, config: ClientConfig):
        self.client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.max_lines = 50000
        self.max_file_size = 450 * 1024 * 1024
        self.max_line_size = 6 * 1024 * 1024
        self.file_index = 1
        self.line_count = 0
        self.current_size = 0
        self.file_source = None
        self.batch_file_path = []

        self.default_config = config
        self.llm_config = self.set_llm_config(config=config)

        self.output_dir = str(os.path.join(pkg.BatchJsonDir))
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def set_llm_config(config: ClientConfig) -> LLMConfig:
        config = ModelAdapter.common_config(config)
        llm_config = LLMConfig(
            model_id="qwen3.5-flash" if config.model is None else config.model,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
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

    def update_llm_config(self, config: ClientConfig) -> LLMConfig:
        merged = ClientConfig.model_validate(
            deep_merge(
                self.default_config.model_dump(),
                config.model_dump(exclude_unset=True)
            )
        )
        return self.set_llm_config(config=merged)

    def get_batch_file_path(self):
        return self.batch_file_path

    def reset_file_line(self):
        self.line_count = 0
        self.current_size = 0

    def open_file(self, filename: str):
        batch_json_path = os.path.join(self.output_dir, f"{filename}.jsonl")
        self.file_source = open(batch_json_path, "w", encoding="utf-8")

        self.batch_file_path.append(batch_json_path)
        self.batch_file_path = list(dict.fromkeys(self.batch_file_path))

    def write_json_to_file(self, json_data: dict[str, Any], filename: str = str(uuid.uuid4())):
        if self.file_source is None or self.file_source.closed:
            self.open_file(filename=f"{filename}_{self.file_index}")

        line = json.dumps(json_data, ensure_ascii=False) + "\n"
        line_bytes = line.encode("utf-8")
        line_size = len(line_bytes)

        if line_size > self.max_line_size:
            raise ValueError(f"JSON line exceeds 6MB: {line_size}")

        if self.line_count > self.max_lines or self.current_size + line_size > self.max_file_size:
            self.file_source.close()
            self.file_index += 1
            self.open_file(filename=f"{filename}_{self.file_index}")
            self.reset_file_line()

        self.file_source.write(line)
        self.line_count += 1
        self.current_size += line_size

    def upload_batch_json(self, file_path: Path) -> str:
        file_object = self.client.files.create(file=file_path, purpose="batch")
        obj = file_object.model_dump()

        id = obj.get("id", None)
        if id is None:
            raise ValueError('id cannot be None')

        logger.info(f"batch_id:{id}")
        batch = self.client.batches.create(
            input_file_id=id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_data = batch.model_dump()
        return batch_data.get("id", "")

    def check_batch_task(self, batch_id: str) -> dict[str, Any]:
        batch = self.client.batches.retrieve(batch_id)
        return batch.model_dump()

    def cancel_back_task(self, batch_id: str) -> dict[str, Any]:
        batch = self.client.batches.cancel(batch_id)
        return batch.model_dump()

    def get_batch_result(self, file_id: str) -> str:
        content = self.client.files.content(file_id=file_id)
        return content.text

    @classmethod
    def preprocess_image(cls, image_bytes: bytes) -> dict[str, Any]:
        image_type = get_image_extension(image_bytes)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_type};base64,{image_base64}",
            },
        }

    @classmethod
    def generate_message(cls, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": content
            }
        ]

    @classmethod
    def generate_text(cls, prompt_list: list[str]) -> list[dict[str, Any]]:
        obj_list = []
        for prompt in prompt_list:
            obj_list.append({"type": "text", "text": prompt})

        return cls.generate_message(content=obj_list)

    def generate_image(self, prompt: str | None = None, images: list[str] | None = None) -> list[dict[str, Any]]:
        content = []
        if prompt is not None and len(prompt) > 0:
            content.append({"type": "text", "text": prompt})

        if images is not None and len(images) > 0:
            for image in images:
                images_bytes = get_file(image)
                image_obj = self.preprocess_image(images_bytes)
                content.append(image_obj)

        return self.generate_message(content=content)

    def generate_batch_json(self, key: str,
                            system: Optional[str] = None,
                            messages: list[dict[str, Any]] | None = None,
                            seed: Optional[int] = None) -> dict[
        str, Any]:

        if messages is None or len(messages) == 0:
            raise ValueError("messages cannot be None")

        if system is not None and len(system) > 0:
            messages = [{"role": "system", "content": system}] + messages

        return {
            "custom_id": key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.llm_config.model_id,
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
                "seed": seed,
                "messages": messages,
                "stream": True
            }
        }

    async def request_llm(self, system: Optional[str] = None,
                          messages: list[dict[str, Any]] | None = None,
                          seed: Optional[int] = None,
                          config: ClientConfig | None = None) -> Result:
        if messages is None or len(messages) == 0:
            raise ValueError("messages cannot be None")

        if system is not None and len(system) > 0:
            messages = [{"role": "system", "content": system}] + messages

        llm_config = self.llm_config
        if config is not None:
            llm_config = self.update_llm_config(config=config)

        result = {
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "model_version": llm_config.model_id,
        }

        try:
            params = {
                "model": llm_config.model_id,
                "messages": messages,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
                "stream": llm_config.stream,
                "parallel_tool_calls": llm_config.parallel_tool_calls,
                "seed": seed,
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

            else:
                if not response.choices:
                    logger.info(f"Usage:{response.usage}")
                else:
                    answer_content = response.choices[0].message.content

                if response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens

            logger.info(f"reasoning_content:{reasoning_content}")

            result["text"] = answer_content
            result["input_tokens"] = input_tokens
            result["output_tokens"] = output_tokens

            return Result(success=True, result=result, error=None)
        except Exception as e:
            return Result(success=False, result=result, error={e})
