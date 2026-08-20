import unittest
from easy_to_markdown.pkg.pdf_to_image import PdfToImage, PdfInfo

class TestReadFile(unittest.TestCase):
    def test_pdf_to_image(self):
        pdfToImage = PdfToImage(PdfInfo(
            pdf_path="/Users/xuanjinliang/PycharmProjects/MinerU/demo/pdfs/demo2.pdf",
            dpi=200,
        ))
        image_info = pdfToImage.save_image()
        print(f"image_info:{image_info}")