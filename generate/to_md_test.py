import unittest
import os
import pkg
import json
from generate import FileParsingResult

from generate.to_md import MarkdownJsonWriter


class ToMarkdown(unittest.TestCase):
    def test_markdown_json_writer(self):
        file_dir = os.path.join(pkg.PdfTempDir, "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c")
        with open(os.path.join(file_dir, "layout_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        file_parsing_result_list = [FileParsingResult.model_validate(item) for item in data]

        md_json_writer = MarkdownJsonWriter(pkg.MDDir)
        md_json_info = md_json_writer.run(file_parsing_result_list)
        print(md_json_info)
