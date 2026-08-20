import os
from easy_to_markdown import pkg
from typing import Literal, Any
from easy_to_markdown.llm_model import LocalModelConfig, ModelInfo
from easy_to_markdown.llm_model.qwen3_vl import QwenTransformersModel, QwenMlxModel


class LocalLLM:
    def __init__(self,
                 temperature: int = 1,
                 max_output_tokens=8192,
                 device: Literal["transformers", "mlx"] = "transformers"):
        self.device = device

        if device == "mlx":
            self.llm_model = QwenMlxModel(config=LocalModelConfig(
                model_path=os.path.join(pkg.ModelDir, "qwen", "Qwen3-VL-4B-Instruct-8bit"),
                temperature=temperature,
                max_output_tokens=max_output_tokens
            ))
        else:
            self.llm_model = QwenTransformersModel(config=LocalModelConfig(
                model_path=os.path.join(pkg.ModelDir, "qwen", "Qwen3-VL-4B-Instruct"),
                temperature=temperature,
                max_output_tokens=max_output_tokens
            ))

    def set_message(self, prompt: str, image_list: list[str], system_info="") -> list[dict[str, Any]]:
        input_prompt = []
        if len(system_info) > 0:
            input_prompt = self.llm_model.generate_message(content=system_info, role="system")

        for image in image_list:
            input_prompt += self.llm_model.preprocess_image(prompt=prompt, images=[image])

        return input_prompt

    async def predict(self, messages: list[list[dict[str, Any]]]) -> list[ModelInfo]:
        results = await self.llm_model.request_vllm(messages=messages)

        return [
            item.result if item.success else ModelInfo()
            for item in results
        ]
