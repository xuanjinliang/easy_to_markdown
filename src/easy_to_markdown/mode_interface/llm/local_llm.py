from typing import Any
from pydantic import BaseModel
from easy_to_markdown.llm_model import ModelInfo
from easy_to_markdown.llm_model import APIModelConfig
from easy_to_markdown.llm_model.service_api import LLMServiceApi


class LocalLLM:
    def __init__(self, config: APIModelConfig):
        self.llm_model = LLMServiceApi(config=config)

    def set_message(self, prompt: str, image_list: list[str], system_info="") -> list[dict[str, Any]]:
        input_prompt = []
        if len(system_info) > 0:
            input_prompt = self.llm_model.generate_message(content=system_info, role="system")

        for image in image_list:
            input_prompt += self.llm_model.preprocess_image(prompt=prompt, images=[image])

        return input_prompt

    async def predict(self,
                      messages: list[list[dict[str, Any]]],
                      schema: type[BaseModel] | None = None) -> list[ModelInfo]:
        if not messages:
            return []

        results = await self.llm_model.request_vllm(messages=messages, schema=schema)

        return [
            item.result if item.success else ModelInfo()
            for item in results
        ]
