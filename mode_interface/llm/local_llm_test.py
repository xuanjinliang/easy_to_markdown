import unittest
import os
import pkg
import json
from pathlib import Path
from generate import FileParsingResult
from llm.local_llm import LocalLLM
from ocr import OCRContent


class TestPPOcr(unittest.IsolatedAsyncioTestCase):
    async def test_local_llm(self):
        path_obj = os.path.join(pkg.PromptDir, "doc_understanding_assistant.md")
        system_info = Path(path_obj).read_text(encoding="utf-8")

        file_dir = os.path.join(pkg.PdfTempDir, "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159")
        with open(os.path.join(file_dir, "layout_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        file_parsing_result_list = [FileParsingResult.model_validate(item) for item in data]

        file_parsing_result_list = file_parsing_result_list[:1]

        local_llm = LocalLLM(system_info=system_info, temperature=0, device="mlx")

        messages = []
        for file_parsing in file_parsing_result_list:
            for block in file_parsing.blocks:
                if block.remove:
                    continue

                llm_crop_path = block.llm_crop_path
                if llm_crop_path is None:
                    continue

                ocr_content = block.ocr_content
                if ocr_content is None:
                    continue

                content = ocr_content.content
                bbox = ocr_content.bbox

                if (not isinstance(content, list) or
                        not isinstance(bbox, list) or
                        len(content) <= 0 or len(bbox) <= 0):
                    continue

                prompt = f"[Ocr Content]\n{content}\n\n[Ocr bbox]\n{bbox}\n"

                message = local_llm.set_message(prompt=prompt, image_list=[llm_crop_path])
                messages.append(message)

        results = await local_llm.predict(messages=messages)
        print(results)
