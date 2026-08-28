import unittest
import os
from easy_to_markdown import pkg
from easy_to_markdown.generate.set_block_content import SetBlockContent
from easy_to_markdown.llm_model import APIModelConfig
from ocr import OCRContent


class TestLayout(unittest.IsolatedAsyncioTestCase):
    model_path = os.path.join(pkg.ModelDir, "qwen_mlx", "Qwen3-VL-4B-Instruct-8bit")

    async def test_set_block_content(self):
        llm_model = SetBlockContent(llm_conf=APIModelConfig(
            model=self.model_path,
            temperature=0,
            base_url="http://localhost:8777/v1",
            api_key="not-needed"
        ))

        image_path = os.path.join(
            pkg.PdfTempDir,
            "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159",
            "llm_crops_img",
            "page_1",
            "19_text.webp"
        )
        ocr_content = OCRContent(
            content=[
                "吴原",
                "Name:"
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

    async def test_image_truncation(self):
        llm_model = SetBlockContent(llm_conf=APIModelConfig(
            model=self.model_path,
            temperature=0,
            base_url="http://localhost:8777/v1",
            api_key="not-needed"
        ))

        image_path1 = os.path.join(
            pkg.PdfTempDir,
            "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c",
            "pdf_image",
            "page_3.webp"
        )

        image_path2 = os.path.join(
            pkg.PdfTempDir,
            "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c",
            "pdf_image",
            "page_4.webp"
        )

        message = llm_model.set_message(
            prompt="Image 1 and Image 2",
            image_list=[image_path1, image_path2],
            system_info_type=2
        )
        results = await llm_model.predict(messages=[message])
        print(results[0].content)
        print(results[0])
