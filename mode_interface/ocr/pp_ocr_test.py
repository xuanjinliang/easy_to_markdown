import unittest
from mode_interface.ocr.pp_ocr import PPOcr, PPOcrVl
from pathlib import Path
import pkg
import os


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
        result = pp_ocr_vl.advanced_recognition(img_list)
        print(result)