import unittest
from mode_interface.ocr.easy_ocr import EasyOcr
from pathlib import Path
import pkg
import os


class TestEasyOcr(unittest.IsolatedAsyncioTestCase):
    async def test_easy_ocr_advanced(self):
        easy_ocr = EasyOcr()

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
        result = await easy_ocr.advanced_recognition(img_list)
        print(result)
