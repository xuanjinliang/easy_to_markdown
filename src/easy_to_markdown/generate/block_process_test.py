import unittest
from easy_to_markdown import pkg
import os
import json
from easy_to_markdown.generate import FileParsingResult
from easy_to_markdown.generate.block_process import set_block_process, remove_repeat_blocks


class TestLayout(unittest.TestCase):
    def test_block_process(self):
        file_dir = os.path.join(pkg.PdfTempDir, "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159")
        with open(os.path.join(file_dir, "layout_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        file_parsing_result_list = [FileParsingResult.model_validate(item) for item in data]

        for item in file_parsing_result_list:
            parsing_result = set_block_process(item.blocks)
            parsing_result = remove_repeat_blocks(parsing_result)
            print(parsing_result)