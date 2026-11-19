from pydantic import BaseModel
from easy_to_markdown import pkg
from easy_to_markdown.pkg.load_config import build_config
from easy_to_markdown.pkg.pdf_to_image import PdfToImage, PdfInfo


class ExecuteConfig(BaseModel):
    pdf_config: PdfInfo


def cli_run_command(args):
    config = build_config(args)

    if config["temp_dir"] is not None:
        pkg.PromptDir = config["temp_dir"]

    if config["output"] is not None:
        pkg.MDDir = config["output"]

    print(config)
    # pdf_dpi = args.pdf_dpi
    # pdf_img_suffix = args.pdf_img_suffix
    #
    # pdfToImage = PdfToImage(PdfInfo(
    #     pdf_path=input_dir,
    #     img_suffix=pdf_img_suffix,
    #     dpi=pdf_dpi,
    # ))
    # image_info = pdfToImage.save_image()
