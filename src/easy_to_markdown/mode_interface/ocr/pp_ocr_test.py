import unittest
from easy_to_markdown.mode_interface.ocr.pp_ocr import PPOcr, PPOcrVl
from pathlib import Path
from easy_to_markdown import pkg
import os

from ocr.pp_ocr_text_detection import PPOcrTextDetection


class TestPPOcr(unittest.IsolatedAsyncioTestCase):
    def test_pp_ocr_advanced(self):
        pp_ocr = PPOcr()

        print(pkg.PdfTempDir)
        dir_path = Path(os.path.join(pkg.PdfTempDir, 'aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159', 'crops_img'))

        image_path1 = os.path.join(dir_path,
                                   'page_2',
                                   "4_table",
                                   "table_crop",
                                   "row_7_0.webp"
                                   )

        image_path2 = os.path.join(dir_path,
                                   'page_2',
                                   "4_table",
                                   "table_crop",
                                   "row_7_1.webp"
                                   )

        img_list = [image_path1, image_path2]
        result = pp_ocr.advanced_recognition(img_list)
        print(result)

    def test_pp_ocr_vl_advanced(self):
        pp_ocr_vl = PPOcrVl()
        dir_path = Path(os.path.join(pkg.PdfTempDir, 'aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c', 'crops_img'))

        image_path1 = os.path.join(dir_path,
                                   'page_1',
                                   "4_text.webp"
                                   )

        image_path3 = os.path.join(dir_path,
                                   'page_1',
                                   '5_table',
                                   "table_crop",
                                   "row_0_0.webp"
                                   )

        image_path4 = os.path.join(dir_path,
                                   'page_1',
                                   '5_table',
                                   "table_crop",
                                   "row_0_1.webp"
                                   )

        img_list = [image_path1, image_path3, image_path4]
        result = pp_ocr_vl.advanced_recognition(img_list)
        print(result)

    def test_pp_ocr_text_detection(self):
        pp_ocr_text_detection = PPOcrTextDetection()
        dir_path = Path(os.path.join(pkg.PdfTempDir, 'aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c', 'crops_img'))
        image_path1 = os.path.join(dir_path, 'page_3', '3_table', 'table_crop', 'row_0_0.webp')
        image_path2 = os.path.join(dir_path, 'page_3', '3_table', 'table_crop', 'row_0_1.webp')
        image_path3 = os.path.join(dir_path, 'page_3', '3_table', 'table_crop', 'row_1_0.webp')
        image_path4 = os.path.join(dir_path, 'page_3', '3_table', 'table_crop', 'row_1_1.webp')
        img_list = [image_path1, image_path2, image_path3, image_path4]
        result = pp_ocr_text_detection.advanced_recognition(img_list)
        print(result)
