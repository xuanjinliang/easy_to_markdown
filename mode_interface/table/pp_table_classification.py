from paddleocr import TableClassification
from paddleocr import TableCellsDetection
from typing import Literal, Any
import os
import pkg
from pkg.common import ensure_dir, chunk_list
from generate import FileParsingResult
from mode_interface.table import TablePosition
from paddlex.inference.models.image_classification.result import TopkResult
from paddlex.inference.models.object_detection.result import DetResult


class PPTableClassification:
    def __init__(self, device: Literal["cpu", "gpu:0"] = "cpu", conf=0.3):
        model_path = os.path.join(pkg.ModelDir, "paddle", "PP-LCNet_x1_0_table_cls")
        model_path_wireless = os.path.join(pkg.ModelDir, "paddle", "RT-DETR-L_wireless_table_cell_det")
        model_path_wired = os.path.join(pkg.ModelDir, "paddle", "RT-DETR-L_wired_table_cell_det")

        ensure_dir([model_path, model_path_wireless, model_path_wired])

        self.model = TableClassification(
            model_name="PP-LCNet_x1_0_table_cls",
            model_dir=model_path,
            device=device,
        )

        self.model_path_wireless = model_path_wireless
        self.model_path_wired = model_path_wired
        self.device = device
        self.conf = conf

    @staticmethod
    def process_item(result: TopkResult) -> str | None:
        label_list = result.get('label_names', [])
        score_list = result.get('scores', 0)

        max_score = 0
        label = None
        for i, item in enumerate(label_list):
            if label is None or score_list[i] > max_score:
                max_score = score_list[i]
                label = item

        return label

    @staticmethod
    def wired_process_item(results: list[DetResult], img_path_list: list[str]) -> list[list[dict[str, Any]]]:
        if not results or not img_path_list:
            return []

        table_info_list: list[list[dict[str, Any]]] = []
        for i, result in enumerate(results):
            boxes = [] if result is None else result.get('boxes', [])
            table_info_list.append(boxes)

        return table_info_list

    def wireless_format(self, img_list: list[str]) -> list[list[dict[str, Any]]]:
        if len(img_list) <= 0:
            return []

        model_wireless = TableCellsDetection(
            model_name="RT-DETR-L_wireless_table_cell_det",
            model_dir=self.model_path_wireless,
            device=self.device,
            threshold=self.conf
        )

        results: list[DetResult] = []
        for img_l in chunk_list(img_list):
            pred = model_wireless.predict(
                input=img_l,
            )
            results += pred

        return self.wired_process_item(results=results, img_path_list=img_list)

    def wired_format(self, img_list: list[str]) -> list[list[dict[str, Any]]]:
        if len(img_list) <= 0:
            return []

        model_wired = TableCellsDetection(
            model_name="RT-DETR-L_wired_table_cell_det",
            model_dir=self.model_path_wired,
            device=self.device,
            threshold=self.conf
        )

        results: list[DetResult] = []
        for img_l in chunk_list(img_list):
            pred = model_wired.predict(
                input=img_l,
            )
            results += pred

        return self.wired_process_item(results=results, img_path_list=img_list)

    def format(self, file_parsing_result: list[FileParsingResult]) -> list[TablePosition]:
        if len(file_parsing_result) == 0:
            return []

        table_list_pos: list[TablePosition] = []
        img_list_pos: list[str] = []

        for i, file_parsing in enumerate(file_parsing_result):
            set_info = TablePosition(img_index=i)
            for j, block in enumerate(file_parsing.blocks):
                if block.block_label == "table":
                    set_info.blocks_index = j
                    if block.crop_path is None:
                        continue
                    set_info.crop_path = block.crop_path
                    table_list_pos.append(set_info)
                    img_list_pos.append(block.crop_path)

        results: list[TopkResult] = []
        for img_list in chunk_list(img_list_pos):
            pred = self.model.predict(
                input=img_list,
            )
            results += pred

        parsing_info_list = [
            self.process_item(result=result)
            for result in results
        ]

        wireless_list: list[TablePosition] = []
        wireless_img_list: list[str] = []
        wired_list: list[TablePosition] = []
        wired_img_list: list[str] = []

        for i, item in enumerate(parsing_info_list):
            if item is None:
                continue
            table_list_pos[i].table_model.append(item)
            crop_path = table_list_pos[i].crop_path

            if item == "wireless_table":
                wireless_list.append(table_list_pos[i])
                wireless_img_list.append(crop_path)
                continue

            if item == "wired_table":
                wired_list.append(table_list_pos[i])
                wired_img_list.append(crop_path)
                continue

        table_position_list = wireless_list + wired_list
        row_and_col_info_list = self.wireless_format(wireless_img_list) + self.wired_format(wired_img_list)

        for i, item in enumerate(table_position_list):
            item.table_content = row_and_col_info_list[i]

        return table_position_list
