import os
from pydantic import BaseModel
from easy_to_markdown import pkg
from typing import Any
from pathlib import Path
from easy_to_markdown.mode_interface.llm.local_llm import LocalLLM, ModelInfo
from easy_to_markdown.llm_model import APIModelConfig
from easy_to_markdown.mode_interface.ocr import OCRContent


class SetBlockContent:
    def __init__(self, llm_conf: APIModelConfig):
        self.local_llm = LocalLLM(config=llm_conf)

    @staticmethod
    def set_system_info(system_info_type: int = 1) -> str:
        prompt_path = ""

        match system_info_type:
            case 1:
                prompt_path = os.path.join(pkg.PromptDir, "doc_assistant.md")
            case 2:
                prompt_path = os.path.join(pkg.PromptDir, "image_truncation_assistant.md")

        if len(prompt_path) <= 0:
            return ""

        return Path(prompt_path).read_text(encoding="utf-8")

    @staticmethod
    def set_ocr_content_prompt(ocr_content: OCRContent | None) -> str | None:
        if ocr_content is None:
            return None

        content = ocr_content.content

        if not isinstance(content, list) or len(content) <= 0:
            return None

        str_content = f'<ocr_text>{"</ocr_text><ocr_text>".join(content).replace("𫊸", "")}</ocr_text>'
        return f"<ocr_content>{str_content}</ocr_content>"

    def set_message(self, prompt: str, image_list: list[str], system_info_type: int) -> list[dict[str, Any]]:
        system_info = self.set_system_info(system_info_type)
        return self.local_llm.set_message(
            prompt=prompt, image_list=image_list, system_info=system_info)

    async def predict(self,
                      messages: list[list[dict[str, Any]]],
                      schema: type[BaseModel] | None = None) -> list[ModelInfo]:
        results = await self.local_llm.predict(messages=messages, schema=schema)

        for result in results:
            if result.content is None:
                result.content = ""
            else:
                for s in ["<empty/>", "<ocr_content>", "</ocr_content>", "<ocr_text>", "</ocr_text>"]:
                    result.content = result.content.replace(s, "")

        return results
