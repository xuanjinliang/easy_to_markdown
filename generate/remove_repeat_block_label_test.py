import unittest
import pkg
import os
import json
from generate import FileParsingResult
from generate.remove_repeat_block_label import RemoveRepeatBlockLabel


class TestLayout(unittest.TestCase):
    def test_repeat_block_label(self):
        file_dir = os.path.join(pkg.PdfTempDir, "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159")
        with open(os.path.join(file_dir, "layout_result_1.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        file_parsing_result_list = [FileParsingResult.model_validate(item) for item in data]

        for item in file_parsing_result_list:
            remove = RemoveRepeatBlockLabel(item.blocks)
            result = remove.run()
            print(
                json.dumps(
                    [item.model_dump() for item in result],
                    ensure_ascii=False,
                    indent=2
                )
            )
