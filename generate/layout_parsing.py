import asyncio
import os
from pydantic import BaseModel, Field
from typing import Literal
from pkg.pdf_to_image import ImageResponse
from concurrent.futures import ThreadPoolExecutor
from mode_interface.layout.pp_doclayout import PPDocLayout
from generate import (FileParsingResult, ParsingResult, TableInfo,
                      TableCellCategory, ColumnsInfo)
from generate.block_process import set_block_process
from pkg.common import ensure_dir, chunk_list, remove_fenced_code_block
from PIL import Image
from pathlib import Path
import cv2
from generate.font_position import layout_labels
from typing import Optional
from mode_interface.table import TablePosition
from generate.table_cell_position import clean_cell_detections, table_cell_category
from mode_interface.table.pp_table_classification import PPTableClassification
from mode_interface.ocr import pp_ocr, OCRContent
import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


class ParsingInfo(BaseModel):
    image_list: list[ImageResponse]
    imgsz: int = 1024
    device: Literal["cpu", "cuda:0"] = "cpu"
    conf: float = Field(default=0.25, gt=0, le=1)
    padding: int = 10
    max_workers: int = Field(default=4, ge=1)
    max_retry: int = 3
    font_scale: float = 0.6
    font_space: int = 4
    border_space: int = 4
    border_line: int = 2
    font_pos_step: int = 10
    thickness: int = 2


class LayoutParsing:
    def __init__(self, parsing_info: ParsingInfo):
        max_workers = parsing_info.max_workers

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.infer_semaphore = asyncio.Semaphore(max_workers)
        self.parsing_info = parsing_info

        # layout
        self.model = PPDocLayout(
            device=parsing_info.device,
            conf=parsing_info.conf
        )

        # table layout
        self.table_model = PPDocLayout(
            device=parsing_info.device,
            conf=0.1,
            layout_merge_bboxes_mode="small"
        )

        # ocr
        self.ocr_model = pp_ocr.PPOcr(device=parsing_info.device)

        # table
        self.table_classification = PPTableClassification()

    def cell_inference(self,
                       image_list: list[ImageResponse],
                       output_dir: str,
                       table_width: float,
                       table_height: float
                       ) -> list[TableCellCategory]:
        if not image_list:
            return []

        category_list: list[TableCellCategory] = []
        parsing_info_list = []

        exec_num = 0
        while exec_num < self.parsing_info.max_retry:
            try:
                parsing_info_list = self.table_model.format(image_list=image_list)
                break
            except Exception as e:
                logger.error(e)
            finally:
                exec_num += 1

        for i, blocks in enumerate(parsing_info_list):
            if not blocks:
                category_list.append(TableCellCategory(
                    category_type="text",
                ))
                continue

            image_info = image_list[i]

            blocks = set_block_process(blocks=blocks)
            category = table_cell_category(blocks=blocks,
                                           cell_image=image_info,
                                           table_w=table_width,
                                           table_h=table_height)

            cell_category = TableCellCategory(
                category_type=category,
            )
            if category == "unknown":
                blocks = self.crop_blocks(image_info=image_info, blocks=blocks, output_dir=output_dir)
                vis_path = self.draw_layout_boxes(image_info=image_info, blocks=blocks, output_dir=output_dir)

                cell_category.parsing_result = FileParsingResult(
                    img_info=image_info,
                    vis_path=vis_path,
                    blocks=blocks,
                )

            category_list.append(cell_category)

        return category_list

    async def table_cell_handle(self,
                                image_list: list[ImageResponse],
                                output_dir: str,
                                table_width: float,
                                table_height: float
                                ) -> list[TableCellCategory]:

        loop = asyncio.get_running_loop()

        async with self.infer_semaphore:
            results = await loop.run_in_executor(
                self.executor,
                self.cell_inference,
                image_list,
                output_dir,
                table_width,
                table_height
            )

        return results

    # 通过PP_doc_layout识别
    async def table_handle_item(self, table_pos: TablePosition) -> tuple[TablePosition, TableInfo | None]:
        img_path = table_pos.crop_path
        table_content = table_pos.table_content

        table_info = clean_cell_detections(img_path=img_path, detections=table_content)

        if table_info is None:
            return table_pos, None

        row_and_col_pos: list[tuple[int, int]] = []
        img_path_list: list[ImageResponse] = []
        for i, rows in enumerate(table_info.table_list):
            for j, columns in enumerate(rows.rows_list):
                row_and_col_pos.append((i, j))

                img_path_list.append(ImageResponse(
                    image_path=columns.image_path,
                    width=columns.width,
                    height=columns.height,
                ))

        output_dir = Path(table_info.img_output_dir).parent
        tasks = []

        for batch in chunk_list(img_path_list, 5):
            task = asyncio.create_task(
                self.table_cell_handle(batch, str(output_dir), table_info.width, table_info.height)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        category_list = [x for r in results for x in r]

        unknown_category_list: list[FileParsingResult] = []
        unknown_row_and_col_pos: list[int] = []
        for index, item in enumerate(category_list):
            i, j = row_and_col_pos[index]
            table_info.table_list[i].rows_list[j].category_type = item.category_type
            if item.category_type == "unknown":
                file_parsing_result = item.parsing_result
                unknown_category_list.append(file_parsing_result)
                unknown_row_and_col_pos.append(index)

        children_file_parsing = await self.table_handle(file_parsing_result=unknown_category_list)
        for index, item in enumerate(children_file_parsing):
            i, j = row_and_col_pos[unknown_row_and_col_pos[index]]
            table_info.table_list[i].rows_list[j].columns_blocks = item

        return table_pos, table_info

    async def table_handle(self, file_parsing_result: list[FileParsingResult]) -> list[FileParsingResult]:
        if not file_parsing_result:
            return file_parsing_result

        list_table_position = self.table_classification.format(file_parsing_result)

        if len(list_table_position) <= 0:
            return file_parsing_result

        file_task = [
            asyncio.create_task(self.table_handle_item(item))
            for item in list_table_position
        ]
        list_tuple = await asyncio.gather(*file_task)

        for table_position, table_info in list_tuple:
            img_index = table_position.img_index
            blocks_index = table_position.blocks_index
            table_model = table_position.table_model

            block = file_parsing_result[img_index].blocks[blocks_index]
            block.table_info = table_info
            block.table_model = table_model

        return file_parsing_result

    def crop_blocks(self,
                    image_info: ImageResponse,
                    blocks: list[ParsingResult],
                    output_dir: str) -> list[ParsingResult]:
        if not blocks:
            return []

        padding = self.parsing_info.padding
        image_path = Path(image_info.image_path)
        output_dir = os.path.join(output_dir, "crops_img", image_path.stem)
        ensure_dir(output_dir)

        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        for block in blocks:
            if block.remove:
                continue
            x1, y1, x2, y2 = block.block_bbox

            crop_x1 = max(0, int(x1) - padding)
            crop_y1 = max(0, int(y1) - padding)
            crop_x2 = min(width, int(x2) + padding)
            crop_y2 = min(height, int(y2) + padding)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                block.crop_path = None
                block.crop_bbox = None
                continue

            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            crop_name = f'{block.block_id}_{block.block_label}.webp'
            crop_path = os.path.join(output_dir, crop_name)

            crop_img.save(crop_path)

            block.crop_path = crop_path
            block.crop_bbox = [crop_x1, crop_y1, crop_x2, crop_y2]

        return blocks

    def draw_layout_boxes(self,
                          image_info: ImageResponse,
                          blocks: list[ParsingResult],
                          output_dir: str) -> Optional[str]:
        if not blocks:
            return None

        output_dir = os.path.join(output_dir, "vis_img")
        ensure_dir(output_dir)

        img = layout_labels(
            image_info=image_info,
            blocks=blocks,
            font_scale=self.parsing_info.font_scale,
            font_space=self.parsing_info.font_space,
            border_space=self.parsing_info.border_space,
            border_line=self.parsing_info.border_line,
            font_pos_step=self.parsing_info.font_pos_step,
            thickness=self.parsing_info.thickness
        )

        if img is None:
            return None

        filename = Path(image_info.image_path).stem
        img_path = os.path.join(output_dir, f"{filename}_layout.webp")
        if cv2.imwrite(img_path, img):
            return img_path

        return None

    def inference(self,
                  image_list: list[ImageResponse],
                  output_dir: str) -> list[FileParsingResult]:
        if len(image_list) == 0:
            return []

        parsing_info_list = []
        exec_num = 0
        while exec_num < self.parsing_info.max_retry:
            try:
                parsing_info_list = self.model.format(image_list=image_list)
                break
            except Exception as e:
                logger.error(e)
            finally:
                exec_num += 1

        file_parsing_list = []

        for i, blocks in enumerate(parsing_info_list):
            image_info = image_list[i]

            blocks = set_block_process(blocks=blocks)
            blocks = self.crop_blocks(image_info=image_info, blocks=blocks, output_dir=output_dir)
            vis_path = self.draw_layout_boxes(image_info=image_info, blocks=blocks, output_dir=output_dir)
            file_parsing_list.append(FileParsingResult(
                img_info=image_info,
                vis_path=vis_path,
                blocks=blocks,
            ))

        return file_parsing_list

    async def layout_handle_item(self,
                                 image_list: list[ImageResponse],
                                 output_dir: str) -> list[FileParsingResult]:
        loop = asyncio.get_running_loop()

        async with self.infer_semaphore:
            results = await loop.run_in_executor(
                self.executor,
                self.inference,
                image_list,
                output_dir,
            )

        return results

    async def ocr_table_info_cell(self, cell_list: list[ColumnsInfo]) -> list[ColumnsInfo]:
        tasks = [
            (i, cell.image_path)
            for i, cell in enumerate(cell_list)
            if cell.image_path is not None
        ]

        loop = asyncio.get_running_loop()
        async with self.infer_semaphore:
            results = await loop.run_in_executor(
                self.executor,
                self.ocr_inference,
                [x[1] for x in tasks],
            )

        for (index, _), ocr_content in zip(tasks, results):
            cell_list[index].ocr_content = ocr_content

        return cell_list

    async def ocr_table_info(self, table_info: TableInfo) -> TableInfo:
        table_list = table_info.table_list
        columns_info_list: list[ColumnsInfo] = []
        for rows in table_list:
            for cols in rows.rows_list:
                if cols.category_type != "unknown":
                    columns_info_list.append(cols)
                else:
                    file_parsing_result = cols.columns_blocks
                    if file_parsing_result is not None:
                        cols.columns_blocks = await self.ocr_handel_file_parsing(
                            file_parsing_result=file_parsing_result)

        if len(columns_info_list) > 0:
            tasks = []
            for batch in chunk_list(columns_info_list, 5):
                task = asyncio.create_task(self.ocr_table_info_cell(batch))
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            columns_results = [x for r in results for x in r]
            for i, result in enumerate(columns_results):
                columns_info_list[i] = result

        return table_info

    def ocr_inference(self, image_list: list[str]) -> list[OCRContent]:
        parsing_info_list = []
        exec_num = 0
        while exec_num < self.parsing_info.max_retry:
            try:
                parsing_info_list = self.ocr_model.advanced_recognition(image_list=image_list)
                break
            except Exception as e:
                logger.error(e)
            finally:
                exec_num += 1

        return parsing_info_list

    async def ocr_handle_item(self,
                              blocks: list[ParsingResult]) -> list[ParsingResult]:

        text_blocks_index: list[tuple[int, str]] = []
        for index, block in enumerate(blocks):
            canonical = block.block_label_type
            match canonical:
                case "image":
                    continue
                case "table":
                    if block.table_info is not None:
                        block.table_info = await self.ocr_table_info(table_info=block.table_info)
                    continue

            crop_img_path = block.crop_path
            if crop_img_path is None:
                continue

            text_blocks_index.append((index, crop_img_path))

        loop = asyncio.get_running_loop()

        async with self.infer_semaphore:
            results = await loop.run_in_executor(
                self.executor,
                self.ocr_inference,
                [x[1] for x in text_blocks_index],
            )

        for (index, _), ocr_content in zip(text_blocks_index, results):
            blocks[index].ocr_content = ocr_content

        return blocks

    async def ocr_handel_file_parsing(self, file_parsing_result: FileParsingResult) -> FileParsingResult:
        blocks = file_parsing_result.blocks
        tasks = []
        for batch in chunk_list(blocks, 5):
            task = asyncio.create_task(self.ocr_handle_item(batch))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        file_parsing_result.blocks = [x for r in results for x in r]
        return file_parsing_result

    async def run(self,
                  image_list: list[ImageResponse],
                  output_dir: str) -> list[FileParsingResult]:
        if not image_list or not output_dir:
            return []

        tasks = []

        for batch in chunk_list(image_list):
            task = asyncio.create_task(self.layout_handle_item(batch, output_dir))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        file_parsing_data = [x for r in results for x in r]

        file_parsing_data = await self.table_handle(file_parsing_data)

        file_parsing_data = [
            await self.ocr_handel_file_parsing(file_parsing_result)
            for file_parsing_result in file_parsing_data
        ]

        return file_parsing_data
