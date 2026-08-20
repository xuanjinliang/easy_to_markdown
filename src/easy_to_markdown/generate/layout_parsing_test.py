import unittest
import os
from easy_to_markdown import pkg
import json
import re
from easy_to_markdown.pkg.pdf_to_image import ImageResponse
from easy_to_markdown.generate import LLMConfig
from easy_to_markdown.generate.layout_parsing import ParsingInfo, LayoutParsing
from pathlib import Path


class TestLayout(unittest.IsolatedAsyncioTestCase):
    async def test_layout_parsing(self):
        output_dir = os.path.join(pkg.PdfTempDir, "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c")

        webp_files = sorted(
            Path(os.path.join(output_dir, "pdf_image")).glob("page_*.webp"),
            key=lambda p: int(re.search(r'\d+', p.stem).group()))

        webp_files = webp_files[:3]
        # webp_files = [webp_files[0]]
        image_list = []
        for i, item in enumerate(webp_files):
            image_list.append(
                ImageResponse(
                    page_index=i,
                    image_path=str(item),
                    width=1700,
                    height=2200
                )
            )

        # output_dir = os.path.join(pkg.PdfTempDir, "test")
        # image_list = [
        #     ImageResponse(
        #         page_index=0,
        #         image_path=os.path.join(output_dir, "pdf_image", "paddleocr_vl_demo.png"),
        #         width=1524,
        #         height=1368
        #     ),
        # ]

        parsing_info = ParsingInfo(
            image_list=image_list,
            llm_conf=LLMConfig(
                device="mlx"
            )
        )
        layout = LayoutParsing(parsing_info=parsing_info)
        results = await layout.run(image_list=image_list, output_dir=output_dir)

        json_data = [r.model_dump() for r in results]
        json_str = json.dumps(
            json_data,
            ensure_ascii=False,
            indent=2
        )
        with open(os.path.join(output_dir, "layout_result.json"), "w", encoding="utf-8") as f:
            f.write(json_str)

    async def test_layout_parsing_1(self):
        output_dir = os.path.join(pkg.PdfTempDir, "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159")

        webp_files = sorted(
            Path(os.path.join(output_dir, "pdf_image")).glob("page_*.webp"),
            key=lambda p: int(re.search(r'\d+', p.stem).group()))

        webp_files = webp_files[:2]
        image_list = []
        for i, item in enumerate(webp_files):
            image_list.append(
                ImageResponse(
                    page_index=i,
                    image_path=str(item),
                    width=1654,
                    height=2340
                )
            )

        parsing_info = ParsingInfo(
            image_list=image_list,
            llm_conf=LLMConfig(
                device="mlx"
            )
        )
        layout = LayoutParsing(parsing_info=parsing_info)
        results = await layout.run(image_list=image_list, output_dir=output_dir)

        json_data = [r.model_dump() for r in results]
        json_str = json.dumps(
            json_data,
            ensure_ascii=False,
            indent=2
        )
        with open(os.path.join(output_dir, "layout_result.json"), "w", encoding="utf-8") as f:
            f.write(json_str)
