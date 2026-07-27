import unittest
import os
import pkg
from mode_interface.layout.yolov import Yolov
from mode_interface.layout.pp_doclayout import PPDocLayout
from generate import ImageResponse


class TestLayoutModel(unittest.TestCase):
    def test_yolov_model(self):
        output_dir = os.path.join(pkg.PdfTempDir, "test")
        image_list = [
            ImageResponse(
                page_index=0,
                image_path=os.path.join(output_dir, "pdf_image", "paddleocr_vl_demo.png"),
                width=1524,
                height=1368
            ),
        ]


        yolov = Yolov()
        result = yolov.format(image_list=image_list)
        print(result)

    def test_pp_doc_layout_model(self):
        output_dir = os.path.join(pkg.PdfTempDir, "test")
        image_list = [
            ImageResponse(
                page_index=0,
                image_path=os.path.join(output_dir, "pdf_image", "paddleocr_vl_demo.png"),
                width=1524,
                height=1368
            ),
        ]


        pp_doc = PPDocLayout()
        result = pp_doc.format(image_list=image_list)
        print(result)