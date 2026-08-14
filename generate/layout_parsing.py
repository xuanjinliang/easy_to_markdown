import asyncio
import os
import numpy as np
import pkg
from pydantic import BaseModel, Field
from typing import Literal
from pkg.pdf_to_image import ImageResponse
from pkg.draw_label import BboxLabel
from concurrent.futures import ThreadPoolExecutor
from mode_interface.layout.pp_doclayout import PPDocLayout
from generate import (FileParsingResult, ParsingResult, TableInfo,
                      TableCellCategory, ColumnsInfo, LLMConfig)
from generate.block_process import set_block_process, remove_repeat_blocks
from pkg.common import ensure_dir, chunk_list
from PIL import Image
from pathlib import Path
import cv2
from generate.font_position import layout_labels, draw_labels_info
from typing import Optional
from mode_interface.table import TablePosition
from generate.table_cell_position import clean_cell_detections, table_cell_category
from generate.table_cell_content import TableCellOCRContent
from mode_interface.table.pp_table_classification import PPTableClassification
from mode_interface.ocr import pp_ocr, OCRContent
from llm.local_llm import LocalLLM
import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


class ParsingInfo(BaseModel):
    image_list: list[ImageResponse]
    device: Literal["cpu", "cuda:0"] = "cpu"
    llm_conf: LLMConfig
    conf: float = Field(default=0.25, gt=0, le=1)
    padding: int = 12
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
        # self.pipeline_lock = asyncio.Lock()
        self.parsing_info = parsing_info

        # layout
        self.model = PPDocLayout(
            device=parsing_info.device,
            conf=parsing_info.conf
        )

        # ocr
        self.ocr_model = pp_ocr.PPOcrVl(device=parsing_info.device)

        # llm_config
        self.llm_config = parsing_info.llm_conf

    def get_llm_model(self, system_info_type: int = 1) -> LocalLLM:

        prompt_path = ""
        match system_info_type:
            case 1:
                prompt_path = os.path.join(pkg.PromptDir, "doc_understanding_assistant.md")
            case 2:
                prompt_path = os.path.join(pkg.PromptDir, "natural_reading_assistant.md")

        system_info = Path(prompt_path).read_text(encoding="utf-8")

        local_llm = LocalLLM(
            system_info=system_info,
            temperature=self.llm_config.temperature,
            max_output_tokens=self.llm_config.max_output_tokens,
            device=self.llm_config.device)

        return local_llm

    @staticmethod
    def set_ocr_content_prompt(ocr_content: OCRContent | None) -> str | None:
        if ocr_content is None:
            return None

        content = ocr_content.content
        bbox = ocr_content.bbox

        if (not isinstance(content, list) or
                not isinstance(bbox, list) or
                len(content) <= 0 or len(bbox) <= 0):
            return None

        return f"[Ocr Content]\n{content}\n\n[Ocr bbox]\n{bbox}\n"

    async def set_table_content(self, table_info: TableInfo) -> TableInfo:
        llm_model = self.get_llm_model(system_info_type=2)

        messages = []
        columns_index: list[tuple[int, int]] = []
        for i, row_info in enumerate(table_info.table_list):
            for j, columns_info in enumerate(row_info.rows_list):
                if columns_info.columns_blocks is not None:
                    results = await self.set_block_content(file_parsing_data=[columns_info.columns_blocks])
                    if len(results) > 0:
                        columns_info.columns_blocks = results[0]
                    continue

                image_path = columns_info.image_path
                if not image_path:
                    continue

                prompt = self.set_ocr_content_prompt(columns_info.ocr_content)
                if prompt is None:
                    continue

                message = llm_model.set_message(prompt=prompt, image_list=[image_path])
                messages.append(message)
                columns_index.append((i, j))

        if len(messages) <= 0:
            return table_info

        results = await llm_model.predict(messages=messages)
        for (i, j), result in zip(columns_index, results):
            row_info = table_info.table_list
            row_info[i].rows_list[j].block_content = result

        return table_info

    async def set_block_content(self, file_parsing_data: list[FileParsingResult]) -> list[FileParsingResult]:
        llm_model = self.get_llm_model(system_info_type=1)

        for file_parsing in file_parsing_data:
            messages = []
            block_index: list[int] = []
            for i, block in enumerate(file_parsing.blocks):
                if block.remove:
                    continue

                if block.table_info is not None:
                    block.table_info = await self.set_table_content(table_info=block.table_info)
                    continue

                llm_crop_path = block.llm_crop_path
                if llm_crop_path is None:
                    continue

                prompt = self.set_ocr_content_prompt(block.ocr_content)
                if prompt is None:
                    continue

                message = llm_model.set_message(prompt=prompt, image_list=[llm_crop_path])
                messages.append(message)
                block_index.append(i)

            if len(messages) <= 0:
                continue

            results = await llm_model.predict(messages=messages)
            for index, result in zip(block_index, results):
                file_parsing.blocks[index].block_content = result

        return file_parsing_data

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
                parsing_info_list = self.model.format(image_list=[item.image_path for item in image_list])
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

            parsing_result = [
                ParsingResult(**block.model_dump())
                for block in blocks
            ]

            image_info = image_list[i]

            parsing_result = set_block_process(blocks=parsing_result)
            category = table_cell_category(blocks=parsing_result,
                                           cell_image=image_info,
                                           table_w=table_width,
                                           table_h=table_height)

            cell_category = TableCellCategory(
                category_type=category,
            )
            if category == "unknown":
                parsing_result = self.crop_blocks(image_info=image_info, blocks=parsing_result, output_dir=output_dir)
                vis_path = self.draw_layout_boxes(image_info=image_info, blocks=parsing_result, output_dir=output_dir)

                cell_category.parsing_result = FileParsingResult(
                    img_info=image_info,
                    vis_path=vis_path,
                    blocks=parsing_result,
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

        # print(f"image_list -> {image_list}")
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

        table_cell_content = TableCellOCRContent(
            device=self.parsing_info.device,
            max_workers=max(1, round(self.parsing_info.max_workers / 2)),
        )
        table_info = await table_cell_content.run(table_info=table_info)

        row_and_col_pos: list[tuple[int, int]] = []
        img_path_list: list[ImageResponse] = []
        for i, rows in enumerate(table_info.table_list):
            for j, columns in enumerate(rows.rows_list):
                row_and_col_pos.append((i, j))

                crop_info = columns.crop_content

                if crop_info is None:
                    continue

                img_path_list.append(ImageResponse(
                    image_path=crop_info.image_path,
                    width=crop_info.width,
                    height=crop_info.height,
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

        if len(unknown_category_list) > 0:
            children_file_parsing = await self.table_handle(file_parsing_result=unknown_category_list)
            for index, item in enumerate(children_file_parsing):
                i, j = row_and_col_pos[unknown_row_and_col_pos[index]]
                table_info.table_list[i].rows_list[j].columns_blocks = item

        return table_pos, table_info

    async def table_handle(self, file_parsing_result: list[FileParsingResult]) -> list[FileParsingResult]:
        if not file_parsing_result:
            return file_parsing_result

        table_classification = PPTableClassification()
        list_table_position = table_classification.format(file_parsing_result)

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
        ocr_output_dir = os.path.join(output_dir, "crops_img", image_path.stem)
        ensure_dir(ocr_output_dir)

        llm_output_dir = os.path.join(output_dir, "llm_crops_img", image_path.stem)
        ensure_dir(llm_output_dir)

        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        for block in blocks:
            if block.remove:
                continue
            x1, y1, x2, y2 = block.block_bbox

            copy_img = img
            for p, output_path in [(padding, ocr_output_dir), (50, llm_output_dir)]:
                if output_path == llm_output_dir:
                    if block.block_label_type in ['image', 'table']:
                        continue

                    draw_img = draw_labels_info(
                        image_info=image_info,
                        bbox_label_list=[BboxLabel(block_label="", block_bbox=block.block_bbox)],
                        font_scale=self.parsing_info.font_scale,
                        font_space=self.parsing_info.font_space,
                        border_space=self.parsing_info.border_space,
                        border_line=self.parsing_info.border_line,
                        font_pos_step=self.parsing_info.font_pos_step,
                        thickness=self.parsing_info.thickness
                    )

                    if draw_img is not None:
                        img_rgb = cv2.cvtColor(
                            draw_img,
                            cv2.COLOR_BGR2RGB
                        )

                        copy_img = Image.fromarray(np.asarray(img_rgb))

                crop_x1 = max(0, int(x1) - p)
                crop_y1 = max(0, int(y1) - p)
                crop_x2 = min(width, int(x2) + p)
                crop_y2 = min(height, int(y2) + p)

                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    block.crop_path = None
                    block.crop_bbox = None
                    continue

                crop_img = copy_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                crop_name = f'{block.block_id}_{block.block_label}.webp'
                crop_path = os.path.join(output_path, crop_name)

                crop_img.save(crop_path)

                if output_path == llm_output_dir:
                    block.llm_crop_path = crop_path
                    block.llM_crop_bbox = [crop_x1, crop_y1, crop_x2, crop_y2]
                else:
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
                parsing_info_list = self.model.format(
                    image_list=[item.image_path for item in image_list]
                )
                break
            except Exception as e:
                logger.error(e)
            finally:
                exec_num += 1

        file_parsing_list = []

        for i, blocks in enumerate(parsing_info_list):
            image_info = image_list[i]

            parsing_result = [
                ParsingResult(**block.model_dump())
                for block in blocks
            ]

            parsing_result = set_block_process(blocks=parsing_result)
            parsing_result = remove_repeat_blocks(blocks=parsing_result)
            parsing_result = self.crop_blocks(image_info=image_info, blocks=parsing_result, output_dir=output_dir)
            vis_path = self.draw_layout_boxes(image_info=image_info, blocks=parsing_result, output_dir=output_dir)
            file_parsing_list.append(FileParsingResult(
                img_info=image_info,
                vis_path=vis_path,
                blocks=parsing_result,
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
            if block.remove:
                continue

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

        if len(text_blocks_index) == 0:
            return blocks

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

        file_parsing_data = await self.set_block_content(file_parsing_data=file_parsing_data)

        return file_parsing_data
