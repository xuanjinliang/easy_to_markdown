import os
from easy_to_markdown import pkg
from easy_to_markdown.pkg.common import ensure_dir
from pathlib import Path
import fitz
import uuid
from pydantic import BaseModel, Field
from PIL import Image


class PdfInfo(BaseModel):
    pdf_path: str
    img_suffix: str = Field(default="webp")
    dpi: int = Field(default=200)


class ImageResponse(BaseModel):
    page_index: int | None = None
    image_path: str
    width: float
    height: float


class PdfResponse(BaseModel):
    pdf_info: PdfInfo
    img_info: list[ImageResponse]
    output_dir: str


class PdfToImage:
    def __init__(self, pdf_info: PdfInfo):
        self.pdf_info = pdf_info
        file_path = Path(pdf_info.pdf_path)

        output_dir = os.path.join(pkg.PdfTempDir, f"{file_path.stem}_{str(uuid.uuid4())}")
        ensure_dir(output_dir)
        self.doc = fitz.open(pdf_info.pdf_path)
        self.file_path = file_path
        self.output_dir = output_dir

    def pdf_page_generator(self):
        doc = self.doc
        dpi = self.pdf_info.dpi
        for page_index in range(len(doc)):
            page = doc[page_index]

            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 4500 or pm.height > 4500:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            yield page_index + 1, pm

    def save_image(self) -> PdfResponse:

        output_image = os.path.join(self.output_dir, "pdf_image")
        ensure_dir(output_image)

        image_info = []
        img_suffix = self.pdf_info.img_suffix
        for page, pix in self.pdf_page_generator():
            output_path = os.path.join(output_image, f"{pkg.PageName}{page}.{img_suffix}")
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img.save(output_path, img_suffix, quality=100)

            image_info.append(ImageResponse(
                page_index=page,
                image_path=output_path,
                width=pix.width,
                height=pix.height,
            ))

        return PdfResponse(
            pdf_info=self.pdf_info,
            img_info=image_info,
            output_dir=self.output_dir,
        )
