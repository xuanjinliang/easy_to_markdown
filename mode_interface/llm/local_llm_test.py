import unittest
import os
import pkg
import json
from pathlib import Path
from generate import FileParsingResult
from llm.local_llm import LocalLLM


class TestPPOcr(unittest.IsolatedAsyncioTestCase):
    async def test_local_llm(self):
        path_obj = os.path.join(pkg.PromptDir, "doc_understanding_assistant.md")
        system_info = Path(path_obj).read_text(encoding="utf-8")

        file_dir = os.path.join(pkg.PdfTempDir, "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159")
        with open(os.path.join(file_dir, "layout_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        file_parsing_result_list = [FileParsingResult.model_validate(item) for item in data]


        local_llm = LocalLLM(temperature=0, device="mlx")



        await local_llm.predict()