import unittest
import os
import pkg
from mode_interface.layout.pp_doclayout import PPDocLayout


class TestLayoutModel(unittest.TestCase):
    def test_pp_doc_layout_model(self):
        output_dir = os.path.join(pkg.PdfTempDir, "test")

        pp_doc = PPDocLayout()
        result = pp_doc.format(image_list=[os.path.join(output_dir, "pdf_image", "paddleocr_vl_demo.png")])
        print(result)
