import os
import pkg
from pkg.common import ensure_dir
from mode_interface.layout import LayoutModel, LayoutResult
from paddleocr import LayoutDetection
from paddlex.inference.models.layout_analysis.result import LayoutAnalysisResult
from typing import Literal


class PPDocLayout(LayoutModel):
    def __init__(self,
                 device: Literal["cpu", "cuda:0"] = "cpu",
                 conf: float = 0.25,
                 layout_merge_bboxes_mode: Literal["large", "small", "union"] | dict = "large",
                 layout_nms: bool = True):
        model_path = os.path.join(pkg.ModelDir, "paddle", "PP-DocLayoutV3")
        ensure_dir(model_path)

        enable_hpi = False
        if device == "cuda:0":
            device = "gpu:0"
            enable_hpi = True

        self.model = LayoutDetection(
            model_name="PP-DocLayoutV3",
            model_dir=model_path,
            device=device,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            threshold=conf,
            layout_nms=layout_nms,
            enable_hpi=enable_hpi
        )

    @staticmethod
    def process_item(result: LayoutAnalysisResult) -> list[LayoutResult]:
        img = result.img
        img_res = img.get('res', None)

        img_width, img_height = 1, 1
        if img_res is not None:
            img_width, img_height = img_res.width, img_res.height

        boxes = result.get('boxes', {})

        blocks = []

        if boxes is None or len(boxes) == 0:
            return blocks

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.get('coordinate', [-1, -1, -1, -1])
            n_x1, n_y1, n_x2, n_y2 = [
                round(x1 / img_width, 6),
                round(y1 / img_height, 6),
                round(x2 / img_width, 6),
                round(y2 / img_height, 6)
            ]
            score = box.get('score', 0)
            label = box.get('label', '')
            order = box.get('order', None)

            # print(f"box.xyxy --> {box.xyxy}, box.cls --> {box.cls}, box.conf --> {box.conf}")

            blocks.append(LayoutResult(
                block_id=i + 1,
                block_label=label,
                block_bbox=[
                    round(float(x1), 2), round(float(y1), 2),
                    round(float(x2), 2), round(float(y2), 2)
                ],
                block_bbox_norm=[
                    round(float(n_x1), 6), round(float(n_y1), 6),
                    round(float(n_x2), 6), round(float(n_y2), 6),
                ],
                score=round(score, 4),
                reading_order=order,
                group_id=i + 1,
            )
            )

        return blocks

    def format(self, image_list: list[str]) -> list[list[LayoutResult]]:
        # print(f"image_list: {image_list}")
        if len(image_list) == 0:
            return []

        results = self.model.predict(
            input=image_list,
            batch_size=1,
        )

        parsing_info_list = [
            self.process_item(result=result)
            for result in results
        ]
        return parsing_info_list
