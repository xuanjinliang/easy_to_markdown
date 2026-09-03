from typing import Literal, Any
import os
import asyncio
from pathlib import Path
from easy_to_markdown.mode_interface.table import TableClassificationInterface
from paddlex.inference.models.image_classification.result import TopkResult
import easy_to_markdown.pkg as pkg
from paddleocr import TableClassification
from easy_to_markdown.pkg.common import chunk_list
from easy_to_markdown.llm_model import APIModelConfig
from easy_to_markdown.llm_model.service_api import LLMServiceApi


class PPTableClassification(TableClassificationInterface):
    def __init__(self, device: Literal["cpu", "gpu:0"] = "cpu"):
        model_path = os.path.join(pkg.ModelDir, "paddle", "PP-LCNet_x1_0_table_cls")

        self.model = TableClassification(
            model_name="PP-LCNet_x1_0_table_cls",
            model_dir=model_path,
            device=device,
        )

    @staticmethod
    def process_item(result: TopkResult) -> str | None:
        label_list = result.get('label_names', [])
        score_list = result.get('scores', 0)

        max_score = 0
        label = None
        for i, item in enumerate(label_list):
            if label is None or score_list[i] > max_score:
                max_score = score_list[i]
                label = item

        return label

    async def handle_item(self, image_list: list[str]) -> list[TopkResult]:
        return self.model.predict(
            input=image_list,
            batch_size=1
        )

    async def predict(self, image_list: list[str]) -> list[str | None]:
        tasks = []

        for batch in chunk_list(image_list, 4):
            task = asyncio.create_task(self.handle_item(batch))
            tasks.append(task)

        top_results = await asyncio.gather(*tasks)

        parsing_data = [x for r in top_results for x in r]

        results: list[str | None] = []
        for item in parsing_data:
            results.append(self.process_item(item))

        return results


class LLMTableClassification(TableClassificationInterface):
    def __init__(self, config: APIModelConfig):
        self.llm_model = LLMServiceApi(config=config)
        path_obj = os.path.join(pkg.PromptDir, "table_classification.md")
        self.system_info = Path(path_obj).read_text(encoding="utf-8")

    async def predict(self, image_list: list[str]) -> list[str | None]:
        if not image_list:
            return []

        input_prompt = []
        for image in image_list:
            prompt = self.llm_model.preprocess_image(prompt=self.system_info, images=[image])
            input_prompt.append(prompt)

        llm_results = await self.llm_model.request_vllm(messages=input_prompt)

        return [
            item.result.content if item.success else None
            for item in llm_results
        ]
