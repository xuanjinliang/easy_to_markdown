import unittest
import os
from easy_to_markdown import pkg
from easy_to_markdown.generate.set_block_content import SetBlockContent
from easy_to_markdown.generate import LLMConfig
from ocr import OCRContent


class TestLayout(unittest.IsolatedAsyncioTestCase):
    async def test_set_block_content(self):
        llm_model = SetBlockContent(llm_conf=LLMConfig(device="mlx"))

        image_path = os.path.join(
            pkg.PdfTempDir,
            "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c",
            "crops_img",
            "page_4",
            "2_table",
            "table_crop",
            "row_1_0.webp"
        )
        ocr_content = OCRContent(
            content=[
                "xx"
            ],
        )
        prompt = llm_model.set_ocr_content_prompt(ocr_content)
        if prompt is None:
            return

        message = llm_model.set_message(
            prompt=prompt,
            image_list=[image_path],
            system_info_type=1
        )
        results = await llm_model.predict(messages=[message])
        print(results[0].content)
        print(results[0])
