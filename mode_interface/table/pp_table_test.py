import unittest
import os
import pkg
import json
from generate import FileParsingResult
from mode_interface.table.pp_table_classification import PPTableClassification


class TestPPTable(unittest.TestCase):
    def test_table_classification(self):
        file_dir = os.path.join(pkg.PdfTempDir, "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c")
        with open(os.path.join(file_dir, "layout_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        file_parsing_result_list = [FileParsingResult.model_validate(item) for item in data]

        file_parsing_result_list = file_parsing_result_list[2:]

        table_classification = PPTableClassification()
        file_parsing_result_list = table_classification.format(
            file_parsing_result=file_parsing_result_list
        )

        print(file_parsing_result_list)
