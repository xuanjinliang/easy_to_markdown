import os
import pkg
from mode_interface.layout import LayoutModel
from doclayout_yolo import YOLOv10, engine
from generate import ParsingResult, ImageResponse
from typing import Literal
from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
import torch

torch.serialization.add_safe_globals(
    [YOLOv10DetectionModel]
)



class Yolov(LayoutModel):
    def __init__(self,
                 imgsz: int = 1024,
                 device: Literal["cpu", "cuda:0"] = "cpu",
                 conf: float = 0.25  # 0 < x <= 1
                 ):
        model_path = os.path.join(
            pkg.ModelDir,
            "doclayout_yolo_cache",
            "doclayout_yolo_docstructbench_imgsz1280_2501.pt")
        self.model = YOLOv10(model_path)
        self.imgsz = imgsz
        self.device = device
        self.conf = conf

    @staticmethod
    def process_item(result: engine.results.Results) -> list[ParsingResult]:
        names = result.names
        boxes = result.boxes

        blocks = []

        if boxes is None or len(boxes) == 0:
            return blocks

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            n_x1, n_y1, n_x2, n_y2 = box.xyxyn[0].tolist()
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = names[cls_id]

            # print(f"box.xyxy --> {box.xyxy}, box.cls --> {box.cls}, box.conf --> {box.conf}")

            blocks.append(ParsingResult(
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
                group_id=i + 1,
            )
            )

        return blocks

    def format(self, image_list: list[ImageResponse]) -> list[list[ParsingResult]]:
        if len(image_list) == 0:
            return []

        results = self.model.predict(
            source=[image_info.image_path for image_info in image_list],
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device
        )

        parsing_info_list = [
            self.process_item(result=result)
            for result in results
        ]
        return parsing_info_list
