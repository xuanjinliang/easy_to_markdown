import os
from easy_to_markdown import pkg
from typing import Any
from pathlib import Path
from easy_to_markdown.mode_interface.llm.local_llm import LocalLLM, ModelInfo
from easy_to_markdown.generate import LLMConfig
from easy_to_markdown.mode_interface.ocr import OCRContent


class SetBlockContent:
    def __init__(self, llm_conf: LLMConfig):

        self.local_llm = LocalLLM(
            temperature=llm_conf.temperature,
            max_output_tokens=llm_conf.max_output_tokens,
            device=llm_conf.device)

    @staticmethod
    def set_system_info(system_info_type: int = 1) -> str:
        prompt_path = ""

        match system_info_type:
            case 1:
                prompt_path = os.path.join(pkg.PromptDir, "doc_understanding_assistant.md")
            case 2:
                prompt_path = os.path.join(pkg.PromptDir, "natural_reading_assistant.md")

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

        str_content = "\n\n".join(content).replace("𫊸", "")
        return f"<ocr_content>\n{str_content}\n</ocr_content>"

    def set_message(self, prompt: str, image_list: list[str], system_info_type: int) -> list[dict[str, Any]]:
        system_info = self.set_system_info(system_info_type)
        return self.local_llm.set_message(
            prompt=prompt, image_list=image_list, system_info=system_info)

    async def predict(self, messages: list[list[dict[str, Any]]]) -> list[ModelInfo]:
        results = await self.local_llm.predict(messages=messages)

        for result in results:
            if result.content is None:
                result.content = ""
            else:
                result.content = result.content.replace("<ocr_content>", "").replace("</ocr_content>", "")

        return results
