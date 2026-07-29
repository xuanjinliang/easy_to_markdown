import os
import pkg
import numpy as np
from pkg.common import ensure_dir
from paddleocr import PaddleOCR
from typing import Literal
from mode_interface.ocr import OCRContent
from paddlex.inference.pipelines.ocr.result import OCRResult


class PPOcr:
    def __init__(self, device: Literal["cpu", "cuda:0"] = "cpu"):
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
