import unittest
import os
from easy_to_markdown import pkg
import json
from easy_to_markdown.generate import FileParsingResult
from easy_to_markdown.mode_interface.table.table_structure_recognition import TableStructureRecognition
from easy_to_markdown.mode_interface.table.table_classification import PPTableClassification, LLMTableClassification
from pathlib import Path
from easy_to_markdown.llm_model import APIModelConfig


class TestPPTable(unittest.IsolatedAsyncioTestCase):
    model_path = os.path.join(pkg.ModelDir, "qwen_mlx", "Qwen3-VL-4B-Instruct-8bit")

    async def test_pp_table_classification(self):
        dir_path = Path(os.path.join(pkg.PdfTempDir, 'aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159', 'crops_img'))

        image_path1 = os.path.join(dir_path,
                                   'page_3',
                                   "3_table.webp"
                                   )

        table_classification = PPTableClassification()
        results = await table_classification.predict([image_path1])
        print(results)

    async def test_LLM_table_classification2(self):
        dir_path = Path(os.path.join(pkg.PdfTempDir, 'aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159', 'crops_img'))

        image_path1 = os.path.join(dir_path,
                                   'page_3',
                                   "3_table.webp"
                                   )

        table_classification = LLMTableClassification(config=APIModelConfig(
            model=self.model_path,
            temperature=0,
            base_url="http://localhost:8777/v1",
            api_key="not-needed"
        ))
        results = await table_classification.predict([image_path1])
        print(results)

    async def test_table_structure_recognition(self):
        file_dir = os.path.join(pkg.PdfTempDir, "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c")
        with open(os.path.join(file_dir, "layout_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        file_parsing_result_list = [FileParsingResult.model_validate(item) for item in data]

        # file_parsing_result_list = file_parsing_result_list[2:]

        table_structure_recognition = TableStructureRecognition(llm_conf=APIModelConfig(
            model=self.model_path,
            temperature=0,
            base_url="http://localhost:8777/v1",
            api_key="not-needed"
        ))
        file_parsing_result_list = await table_structure_recognition.format(
            file_parsing_result=file_parsing_result_list
        )

        print(file_parsing_result_list)
