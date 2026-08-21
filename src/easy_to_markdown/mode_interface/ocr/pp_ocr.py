import os
from easy_to_markdown import pkg
import numpy as np
from easy_to_markdown.pkg.common import ensure_dir
from paddleocr import PaddleOCR, PaddleOCRVL
from typing import Literal
from easy_to_markdown.mode_interface.ocr import OCRContent, OcrInterface
from paddlex.inference.pipelines.ocr.result import OCRResult
from paddlex.inference.pipelines.paddleocr_vl.result import PaddleOCRVLResult


class PPOcr(OcrInterface):
    def __init__(self,
                 device: Literal["cpu", "cuda:0"] = "cpu",
                 text_thresh: float = 0.6):
        model_path = os.path.join(pkg.ModelDir, "paddle")
        ensure_dir(model_path)

        enable_hpi = None
        if device == "cuda:0":
            device = "gpu:0"
            enable_hpi = True

        self.model = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            textline_orientation_model_dir=os.path.join(model_path, "PP-LCNet_x1_0_textline_ori"),
            text_detection_model_dir=os.path.join(model_path, "PP-OCRv6_medium_det"),
            text_recognition_model_dir=os.path.join(model_path, "PP-OCRv6_medium_rec"),
            device=device,
            enable_hpi=enable_hpi,
            textline_orientation_batch_size=1,
            text_recognition_batch_size=1,
            text_rec_score_thresh=text_thresh,
            return_word_box=True
        )

    @staticmethod
    def process_item(result: OCRResult) -> OCRContent:
        input_path = result.get("input_path", "")
        model_settings = result.get("model_settings", {})
        text_det_params = result.get("text_det_params", {})
        textline_orientation_angles = result.get("textline_orientation_angles", [])
        text_rec_score_thresh = result.get("text_rec_score_thresh", 0.0)
        content = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        rec_boxes = result.get("rec_boxes", [])

        bbox = rec_boxes
        if isinstance(rec_boxes, np.ndarray):
            bbox = rec_boxes.tolist()

        return OCRContent(
            input_path=input_path,
            model_settings=model_settings,
            text_det_params=text_det_params,
            textline_orientation_angles=textline_orientation_angles,
            text_rec_score_thresh=text_rec_score_thresh,
            content=content,
            scores=scores,
            bbox=bbox,
        )

    def advanced_recognition(self, image_list: list[str]) -> list[OCRContent]:
        if not image_list:
            return []

        results = self.model.predict(image_list)

        parsing_info_list = [
            self.process_item(result=result)
            for result in results
        ]

        return parsing_info_list


class PPOcrVl(OcrInterface):
    def __init__(self,
                 device: Literal["cpu", "cuda:0"] = "cpu"):
        model_path = os.path.join(pkg.ModelDir, "paddle")
        ensure_dir(model_path)

        enable_hpi = None
        if device == "cuda:0":
            device = "gpu:0"
            enable_hpi = True

        self.pipeline = PaddleOCRVL(
            vl_rec_model_name="PaddleOCR-VL-1.6-0.9B",
            vl_rec_model_dir=os.path.join(model_path, "PaddleOCR-VL-1.6"),
            use_layout_detection=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            merge_layout_blocks=False,
            device=device,
            enable_hpi=enable_hpi
        )

    @staticmethod
    def process_item(result: PaddleOCRVLResult) -> OCRContent:
        input_path = result.get("input_path", "")
        model_settings = result.get("model_settings", {})
        parsing_res_list = result.get("parsing_res_list", [])
        content: list[str] = []
        bbox: list[list[float]] = []
        for item in parsing_res_list:
            content.append(item.content)
            bbox.append(item.bbox)

        return OCRContent(
            input_path=input_path,
            model_settings=model_settings,
            content=content,
            bbox=bbox,
        )

    def advanced_recognition(self, image_list: list[str]) -> list[OCRContent]:
        if not image_list:
            return []

        results = self.pipeline.predict(image_list)

        parsing_info_list = [
            self.process_item(result=result)
            for result in results
        ]

        return parsing_info_list
