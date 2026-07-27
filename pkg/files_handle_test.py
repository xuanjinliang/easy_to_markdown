import unittest

import re, json
from pkg.files_handle import file_sha256
from pathlib import Path

def to_plain_text(s: str) -> str:
    s = json.loads(s)
    s = re.sub(r"^```latex", "", s)
    s = re.sub(r"```$", "", s)

    return s.strip()

def to_plain_html(s: str) -> str:
    s = json.loads(s)
    s = re.sub(r"^```html", "", s)
    s = re.sub(r"```$", "", s)

    return s.strip()

class TestReadFile(unittest.TestCase):
    def test_get_file_sha256(self):
        file_path = "/Users/xuanjinliang/PycharmProjects/autogen_agent/pdf_temp/pdf/aws_2023.pdf"

        sha256_id = file_sha256(file_path)
        print(f"sha256_id: {sha256_id}")

    def test_to_plain_text(self):
        file_path = "/Users/xuanjinliang/PycharmProjects/pdf_to_markdown_sdk/pdf_to_md_sdk/pdf_temp/ocr/49d22f2d-9db4-45ff-b8e5-73a5ef8b9d09/page_2.txt"
        markdown_text = Path(file_path).read_text(encoding="utf-8")
        plain_text = to_plain_text(markdown_text)

        output_path = "/Users/xuanjinliang/PycharmProjects/pdf_to_markdown_sdk/pdf_to_md_sdk/pdf_temp/ocr/49d22f2d-9db4-45ff-b8e5-73a5ef8b9d09/page_2_test.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(plain_text)

    def test_to_plain_html(self):
        file_path = "/Users/xuanjinliang/PycharmProjects/pdf_to_markdown_sdk/pdf_to_md_sdk/pdf_temp/ocr/70da3eb3-3523-4f83-9d0e-58a09121700b/page_3.txt"
        markdown_text = Path(file_path).read_text(encoding="utf-8")
        plain_text = to_plain_html(markdown_text)

        output_path = "/Users/xuanjinliang/PycharmProjects/pdf_to_markdown_sdk/pdf_to_md_sdk/pdf_temp/ocr/70da3eb3-3523-4f83-9d0e-58a09121700b/page_3.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(plain_text)
