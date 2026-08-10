import os
import pkg
from typing import Literal
from llm_model import LocalModelConfig, ModelInfo
from llm_model.qwen3_vl import QwenTransformersModel, QwenMlxModel
from mode_interface.llm import LLMContent


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

    async def predict(self, prompt: str, image_list: list[str]) -> list[LLMContent]:

        input_list = []
        for image in image_list:
            input_prompt = self.llm_model.preprocess_image(prompt=prompt, images=[image])
            input_list.append(input_prompt)

        results = await self.llm_model.request_vllm(messages=input_list)

        return [
            LLMContent(input_path=image_path, content=ModelInfo(**item.result))
            for image_path, item in zip(input_list, results)
        ]
