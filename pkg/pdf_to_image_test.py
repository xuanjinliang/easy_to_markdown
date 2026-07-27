import unittest
from pkg.pdf_to_image import PdfToImage, PdfInfo

class TestReadFile(unittest.TestCase):
    def test_pdf_to_image(self):
        pdfToImage = PdfToImage(PdfInfo(
            pdf_path="/Users/xuanjinliang/Downloads/账单合同/AWS cur/合同/MOBVISTA INTERNATIONAL TECHNOLOGY LIMITED-AWS Private Pricing Addendum-26150.1-20241201-Executable1-1_2024.pdf",
            dpi=200,
        ))
        image_info = pdfToImage.save_image()
        print(f"image_info:{image_info}")