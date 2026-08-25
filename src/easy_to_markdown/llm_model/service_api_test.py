import unittest
import os
from easy_to_markdown import pkg
from pydantic import BaseModel
from easy_to_markdown.llm_model.service_api import LLMServiceApi
from easy_to_markdown.llm_model import APIModelConfig


class Description(BaseModel):
    text: str


class TestLayoutModel(unittest.IsolatedAsyncioTestCase):
    async def test_service_api_schema(self):
        model_path = os.path.join(pkg.ModelDir, "qwen_mlx", "Qwen3-VL-4B-Instruct-8bit")
        llm_service_api = LLMServiceApi(config=APIModelConfig(
            model=model_path,
            temperature=0,
            base_url="http://localhost:8777/v1",
            api_key="not-needed"
        ))

        image_list = [
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/1_header.webp"),
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/2_doc_title.webp")
        ]

        input_list = []
        for image in image_list:
            input_prompt = llm_service_api.preprocess_image(prompt="Describe this image.",
                                                           images=[image])
            input_list.append(input_prompt)

        results = await llm_service_api.request_vllm(messages=input_list, schema=Description)
        print(results)
