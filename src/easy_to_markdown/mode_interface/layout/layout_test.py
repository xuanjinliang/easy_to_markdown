import unittest
import os
from easy_to_markdown import pkg
from easy_to_markdown.mode_interface.layout.pp_doclayout import PPDocLayout
from easy_to_markdown.pkg.coordinate_overlap import get_overlap_result_np


class TestLayoutModel(unittest.TestCase):
    def test_pp_doc_layout_model(self):
        output_dir = os.path.join(pkg.PdfTempDir, "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159")

        pp_doc = PPDocLayout()
        result = pp_doc.format(image_list=[
            os.path.join(
                output_dir,
                "pdf_image",
                "page_1.webp")])

        for page_layout in result:
            coordinate = [item.block_bbox for item in page_layout]
            coor_results = get_overlap_result_np(coordinate=coordinate, no_duplicate=True)
            print(coor_results)




