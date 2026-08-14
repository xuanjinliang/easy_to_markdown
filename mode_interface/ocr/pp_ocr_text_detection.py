import os
import pkg
import numpy as np
from paddleocr import TextDetection
from typing import Literal
from pkg.common import ensure_dir
from paddlex.inference.models.text_detection.result import TextDetResult


class PPOcrTextDetection:
    def __init__(self, device: Literal["cpu", "cuda:0"] = "cpu", box_thresh: float = 0.5):
        model_path = os.path.join(pkg.ModelDir, "paddle")
        ensure_dir(model_path)

        enable_hpi = None
        if device == "cuda:0":
            device = "gpu:0"
            enable_hpi = True

        self.model = TextDetection(
            model_name="PP-OCRv6_medium_det",
            model_dir=os.path.join(model_path, "PP-OCRv6_medium_det"),
            device=device,
            box_thresh=box_thresh,
            enable_hpi=enable_hpi
        )

    @staticmethod
    def process_item(result: TextDetResult) -> list[list[float]]:
        dt_polys = result.get('dt_polys', [])

        if isinstance(dt_polys, np.ndarray):
            result_np = np.concatenate([
                dt_polys[:, :, 0].min(axis=1, keepdims=True),
                dt_polys[:, :, 1].min(axis=1, keepdims=True),
                dt_polys[:, :, 0].max(axis=1, keepdims=True),
                dt_polys[:, :, 1].max(axis=1, keepdims=True),
            ], axis=1)
            return result_np.tolist()

        return dt_polys

    def advanced_recognition(self, image_list: list[str]) -> list[list[list[float]]]:
        if not image_list:
            return []

        results = self.model.predict(input=image_list)

        parsing_info_list = [
            self.process_item(result=result)
            for result in results
        ]

        return parsing_info_list
