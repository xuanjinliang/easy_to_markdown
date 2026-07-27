import unittest
from generate.download_model import download_model
from pkg.common import ensure_dir
import os
import pkg
from paddleocr import PaddleOCRVL, PPStructureV3, LayoutDetection


class TestDownloadModel(unittest.TestCase):
    def test_download(self):
        download_model(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )

    def test_paddle_ocrvl(self):
        pipeline = PaddleOCRVL(device="cpu")

        image_path = os.path.join(pkg.PdfTempDir, "test", "pdf_image", "paddleocr_vl_demo.png")
        output_dir = os.path.join(pkg.PdfTempDir, "paddle_ocrvl")

        ensure_dir(output_dir)

        output = pipeline.predict(image_path)
        for res in output:
            res.print()  ## 打印预测的结构化输出
            res.save_to_json(save_path=output_dir)
            res.save_to_markdown(save_path=output_dir)

    def test_ppstructurev3(self):
        pipeline = PPStructureV3(
            layout_detection_model_name="PP-DocLayout_plus-L",
            device="cpu"
        )

        image_path = os.path.join(pkg.PdfTempDir,
                                  "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159",
                                  "pdf_image",
                                  "page_1.webp")
        output_dir = os.path.join(pkg.PdfTempDir, "paddle_structure_v3")

        ensure_dir(output_dir)

        output = pipeline.predict(image_path)
        for res in output:
            res.print()  ## 打印预测的结构化输出
            res.save_to_json(save_path=output_dir)
            res.save_to_markdown(save_path=output_dir)

    def test_pp_layout_detection(self):
        pipeline = LayoutDetection(
            model_name="PP-DocLayoutV3",
            device="cpu"
        )

        image_path = os.path.join(pkg.PdfTempDir,
                                  "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159",
                                  "pdf_image",
                                  "page_1.webp")
        output_dir = os.path.join(pkg.PdfTempDir, "paddle_layout_detection")

        ensure_dir(output_dir)

        output = pipeline.predict(image_path)
        for res in output:
            res.print()  ## 打印预测的结构化输出
            res.save_to_json(save_path=output_dir)
