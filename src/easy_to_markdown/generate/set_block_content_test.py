import unittest
import os
from pydantic import BaseModel
from easy_to_markdown import pkg
from easy_to_markdown.generate.set_block_content import SetBlockContent
from easy_to_markdown.llm_model import APIModelConfig
from ocr import OCRContent


class Description(BaseModel):
    truncated: bool
    reason: str


class TestLayout(unittest.IsolatedAsyncioTestCase):
    model_path = os.path.join(pkg.ModelDir, "qwen_mlx", "Qwen3-VL-8B-Instruct-8bit")

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
            "crops_img",
            "page_10",
            "7_table",
            "table_crop",
            "row_1_1.webp"
        )
        ocr_content = OCRContent(
            content=[
                "For your use of Designated Amazon EC2 Spot Instances under Eligible",
                "Accounts in the applicable Service-Specific Regions qualified as",
                "Selected Amazon EC2 Spot Instances, the applicable Service-Specific",
                "Rate from the Qualifying Month will be $0.0075/vCPU/hour. For",
                "clarity, for your use of Designated Amazon EC2 Spot Instances under",
                "Eligible Accounts in the applicable Service-Specific Regions not",
                "qualified as Selected Amazon EC2 Spot Instances, the applicable",
                "Service-Specific Rate will continue to be $0.008/vCPU/hour.",
                "Qualifying Month means the month AWS receives an Selected",
                "Amazon EC2 Spot Instances Attestation provided such attestation is",
                "received before the 20th day of such month; otherwise the second",
                "month after receipt of such Selected Amazon EC2 Spot Instances",
                "Attestation."
            ],
        )
        prompt = llm_model.set_ocr_content_prompt(ocr_content)
        if prompt is None:
            return

        message = llm_model.set_message(
            prompt=prompt,
            image_list=[image_path],
            system_info_type=3
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

        # image_path1 = os.path.join(
        #     pkg.PdfTempDir,
        #     "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c",
        #     "pdf_image",
        #     "page_3.webp"
        # )
        #
        # image_path2 = os.path.join(
        #     pkg.PdfTempDir,
        #     "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c",
        #     "pdf_image",
        #     "page_4.webp"
        # )

        image_path1 = os.path.join(
            pkg.PdfTempDir,
            "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159",
            "pdf_image",
            "page_3.webp"
        )

        image_path2 = os.path.join(
            pkg.PdfTempDir,
            "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159",
            "pdf_image",
            "page_4.webp"
        )

        messages = llm_model.set_diff_prompt_image_message(
            prompt_image_list=[("Image 1", [image_path1]), ("Image 2", [image_path2])],
            system_info_type=2
        )
        results = await llm_model.predict(messages=[messages], schema=Description)
        print(results[0].content)
        print(results[0])
