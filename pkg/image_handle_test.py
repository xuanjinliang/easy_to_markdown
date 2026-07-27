import unittest
from pkg.files_handle import get_file
from pkg.image_handle import get_image_extension

class TestReadFile(unittest.TestCase):
    def test_str_to_seed(self):

        image_types = get_file("/Users/xuanjinliang/PycharmProjects/autogen_agent/pdf_temp/images/426bcb12-8ba0-40bb-9558-99e6ce88a655/page_1.webp")

        img_type = get_image_extension(image_types)
        print(img_type)
