import unittest
import os
from easy_to_markdown import pkg
import json
from easy_to_markdown.generate import FileParsingResult
from easy_to_markdown.generate.to_md import MarkdownJsonWriter, MarkdownFileResult, MarkdownWriter


class ToMarkdown(unittest.TestCase):
    def test_markdown_json_writer(self):
        file_dir = os.path.join(pkg.PdfTempDir, "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c")
        with open(os.path.join(file_dir, "layout_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        file_parsing_result_list = [FileParsingResult.model_validate(item) for item in data]

        md_json_writer = MarkdownJsonWriter(pkg.MDDir)
        md_json_info = md_json_writer.run(file_parsing_result_list)
        print(md_json_info)

        json_data = [r.model_dump() for r in md_json_info]
        json_str = json.dumps(
            json_data,
            ensure_ascii=False,
            indent=2
        )

        with open(os.path.join(pkg.MDDir, "md_result.json"), "w", encoding="utf-8") as f:
            f.write(json_str)

    def test_output_markdown(self):
        with open(os.path.join(pkg.MDDir, "md_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        list_markdown_file_result = [MarkdownFileResult.model_validate(item) for item in data]

        with MarkdownWriter(file_path=os.path.join(pkg.MDDir, "md_result.md")) as md_writer:
            for file_result in list_markdown_file_result:
                for item in file_result.children:
                    content = item.block_image_content if item.block_image_content is not None else item.block_content
                    md_writer.write(content)

