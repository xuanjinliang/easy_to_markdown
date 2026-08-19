import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from PIL import Image
from generate import TableInfo, ContainerInfoPath
from pkg.common import ensure_dir, chunk_list
from mode_interface.ocr.pp_ocr_text_detection import PPOcrTextDetection

import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


class TableCellOCRContent:
    def __init__(self, device: Literal["cpu", "cuda:0"] = "cpu",
                 max_retry: int = 3,
                 max_workers: int = 4):
        self.text_detection_model = PPOcrTextDetection(device=device)

        self.max_retry = max_retry
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.infer_semaphore = asyncio.Semaphore(max_workers)

    def cell_inference(self, image_list: list[str]) -> list[list[list[float]]]:

        if not image_list:
            return []

        exec_num = 0
        while exec_num < self.max_retry:
            try:
                parsing_info_list = self.text_detection_model.advanced_recognition(image_list=image_list)
                return parsing_info_list
            except Exception as e:
                logger.error(e)
            finally:
                exec_num += 1

        return []

    async def cell_handle(self,
                          image_list: list[str]) -> list[list[list[float]]]:

        loop = asyncio.get_running_loop()

        async with self.infer_semaphore:
            results = await loop.run_in_executor(
                self.executor,
                self.cell_inference,
                image_list,
            )

        return results

    async def run(self, table_info: TableInfo) -> TableInfo:
        row_and_col_pos: list[tuple[int, int]] = []
        img_path_list: list[str] = []
        for i, rows in enumerate(table_info.table_list):
            for j, columns in enumerate(rows.rows_list):
                row_and_col_pos.append((i, j))
                img_path_list.append(columns.image_path)

        output_dir = Path(table_info.img_output_dir).parent
        output_dir = os.path.join(output_dir, "cell_crop_info")
        ensure_dir(output_dir)

        tasks = []

        for batch in chunk_list(img_path_list, 5):
            task = asyncio.create_task(
                self.cell_handle(batch)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        ocr_content_list = [x for r in results for x in r]

        for (i, j), ocr_content in zip(row_and_col_pos, ocr_content_list):
            cell = table_info.table_list[i].rows_list[j]
            image_path = Path(cell.image_path)
            # print(f"image_path -> {str(image_path)}")

            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size

            if len(ocr_content) <= 0:
                continue

            x1, y1, x2, y2 = map(float, ocr_content[0])

            # print(f"bbox --> {ocr_content}")

            for bbox in ocr_content[1:]:
                b_x1, b_y1, b_x2, b_y2 = map(float, bbox)

                x1 = min(x1, b_x1)
                y1 = min(y1, b_y1)
                x2 = max(x2, b_x2)
                y2 = max(y2, b_y2)

            crop_x1 = max(0, int(x1) - 4)
            crop_y1 = max(0, int(y1) - 4)
            crop_x2 = min(img_w, int(x2) + 4)
            crop_y2 = min(img_h, int(y2) + 4)

            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_path = os.path.join(output_dir, image_path.name)
            crop_img.save(crop_path)

            crop_info = ContainerInfoPath(
                image_path=crop_path,
                width=img_w,
                height=img_h,
                bbox=[crop_x1, crop_y1, crop_x2, crop_y2]
            )
            cell.crop_info = crop_info

            # print(f"crop_bbox -> {[crop_x1, crop_y1, crop_x2, crop_y2]}")

        return table_info
