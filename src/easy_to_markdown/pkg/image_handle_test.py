import unittest
import os
from easy_to_markdown import pkg
from easy_to_markdown.pkg.files_handle import get_file
from src.easy_to_markdown.pkg.image_handle import get_image_extension


class TestReadFile(unittest.TestCase):
    def test_str_to_seed(self):
        image_types = get_file(os.path.join(pkg.PdfTempDir, "/images/426bcb12-8ba0-40bb-9558-99e6ce88a655/page_1.webp"))

        img_type = get_image_extension(image_types)
        print(img_type)
