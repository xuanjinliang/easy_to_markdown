from easy_to_markdown.llm_model import APIModelConfig
from easy_to_markdown.pkg.common import ensure_dir
import os
from pathlib import Path
from uuid import uuid4
import shutil
from easy_to_markdown.pkg.enum_class import BlockType
from easy_to_markdown.pkg.format_table import Table, TableCell
from easy_to_markdown.generate import (FileParsingResult, ParsingResult, MarkdownInfo, MarkdownFileResult,
                                       ModelInfo, TableInfo, ImgMergeInfo)
from easy_to_markdown.generate.set_block_content import SetBlockContent
from pydantic import BaseModel


class Description(BaseModel):
    truncated: bool
    reason: str


class MarkdownWriter:
    def __init__(self, file_path: str):
        self.md_file = open(file_path, "w", encoding="utf-8")

    def write(self, markdown_info: MarkdownInfo):
        content = markdown_info.block_image_content if (
                markdown_info.block_image_content is not None) else markdown_info.block_content
        self.md_file.write(content)
        self.md_file.flush()

    def write_list(self, list_markdown_info: list[MarkdownInfo]):
        if len(list_markdown_info) == 0:
            return

        for markdown_info in list_markdown_info:
            self.write(markdown_info)

    def close(self):
        if not self.md_file.closed:
            self.md_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class MarkdownJsonWriter:
    def __init__(self,
                 output_dir: str,
                 tolerance: float = 5.0,
                 ignore_labels: list[str] | None = None,
                 ignore_header: bool = True,
                 ignore_footer: bool = True,
                 is_merge: bool = True,
                 llm_conf: APIModelConfig | None = None):
        self.output_dir = output_dir
        ensure_dir(output_dir)

        self.tolerance = tolerance

        if is_merge:
            ignore_header = True
            ignore_footer = True

        ignore_labels = [] if ignore_labels is None else ignore_labels
        if ignore_header:
            ignore_labels += [BlockType.HEADER, BlockType.HEADER_IMAGE]

        self.ignore_footer_label = [BlockType.FOOTER, BlockType.FOOTER_IMAGE,
                                    BlockType.FOOTNOTE] if ignore_footer else []

        self.ignore_labels = list(dict.fromkeys(ignore_labels))

        self.is_merge = is_merge
        self.llm_model = SetBlockContent(llm_conf=llm_conf) if llm_conf is not None else None

    def set_content(self, block: ParsingResult) -> MarkdownInfo | None:
        if len(self.ignore_labels) > 0 and block.block_label in self.ignore_labels:
            return None

        markdown_info = MarkdownInfo(
            block_id=block.block_id,
            block_label=block.block_label,
            block_bbox=block.block_bbox,
            block_label_type=block.block_label_type,
            level=block.level,
            block_content="",
            block_image_content=None
        )

        label_type = block.block_label_type

        match label_type:
            case "title":
                if isinstance(block.block_content, ModelInfo):
                    markdown_info.block_content = f"{'#' * block.level} {block.block_content.content}\n\n"
            case "image":
                if block.crop_path is not None and os.path.exists(block.crop_path):
                    markdown_info.block_image_content = self.set_image_content(block.crop_path)
            case "table":
                if isinstance(block.table_info, TableInfo):
                    markdown_info.block_content = self.set_table_content(block.table_info)
            case _:
                if isinstance(block.block_content, ModelInfo):
                    markdown_info.block_content = f"{block.block_content.content}\n\n"

        return markdown_info

    def set_only_table_cell(self, table_info: TableInfo) -> list[list[TableCell]]:
        if len(table_info.table_list) == 0:
            return []

        rows_list = []
        for rows in table_info.table_list:
            table_cell: list[TableCell] = []
            table_rows_extra = []
            for cell in rows.rows_list:
                if cell.columns_blocks is not None:
                    blocks = cell.columns_blocks.blocks
                    content = ""

                    if len(blocks) == 1 and blocks[0].block_label_type == "table":
                        cell_table_info = blocks[0].table_info
                        if cell_table_info is not None:
                            row_cell_list = self.set_only_table_cell(table_info=cell_table_info)

                            last_bbox = None
                            if len(table_cell) > 0:
                                last_bbox = table_cell[-1].bbox

                            if last_bbox is not None and len(last_bbox) == 4:
                                for rows_i in row_cell_list:
                                    for cell in rows_i:
                                        bbox = cell.bbox
                                        bbox[0] += last_bbox[2]
                                        bbox[2] += last_bbox[2]
                                        bbox[1] += last_bbox[1]
                                        bbox[3] += last_bbox[1]
                                        cell.bbox = bbox

                            if len(row_cell_list) > 0:
                                table_cell += row_cell_list[0]
                                table_rows_extra.append(row_cell_list[1:])
                            continue

                    for block in blocks:
                        markdown_info = self.set_content(block)
                        if markdown_info is None:
                            continue

                        content += markdown_info.block_content

                    table_cell.append(TableCell(html=content, bbox=cell.bbox))
                    continue

                content = cell.block_content.content if cell.block_content is not None else ""
                table_cell.append(TableCell(text=content, bbox=cell.bbox))

            if len(table_cell) > 0:
                rows_list.append(table_cell)

            if len(table_rows_extra) > 0:
                for item_list in table_rows_extra:
                    for row in item_list:
                        rows_list.append(row)

        return rows_list

    def set_table_content(self, table_info: TableInfo) -> str:
        if len(table_info.table_list) == 0:
            return ""

        table = Table(width=table_info.width, height=table_info.height)

        rows_list = self.set_only_table_cell(table_info)
        for table_cell in rows_list:
            table.add_row(cells=table_cell)

        table.calculate_spans(tolerance=self.tolerance)
        return table.to_html() + "\n\n"

    def set_image_content(self, image_path: str) -> str:
        original = Path(image_path)
        dst_dir = os.path.join(self.output_dir, "images")
        ensure_dir(dst_dir)

        dst = Path(dst_dir) / f"{uuid4()}{original.suffix}"

        shutil.copy2(original, dst)

        image_path = str(dst.relative_to(self.output_dir))

        return f'<img src="{image_path}" alt="Image" />\n\n'

    def generate_blocks(self, blocks: list[ParsingResult]) -> list[MarkdownInfo]:
        blocks_info_list: list[MarkdownInfo] = []
        for block in blocks:
            if len(self.ignore_footer_label) > 0 and block.block_label in self.ignore_footer_label:
                break

            if block.remove:
                continue

            md_info = self.set_content(block)
            if md_info is None or (len(md_info.block_content) == 0 and md_info.block_image_content is None):
                continue

            blocks_info_list.append(md_info)

        return blocks_info_list

    async def marge_paper(self, markdown_file_result: MarkdownFileResult) -> MarkdownFileResult:
        if self.llm_model is None:
            return markdown_file_result

        img_info = markdown_file_result.img_info
        if len(img_info) <= 1:
            return markdown_file_result

        marge_split_list = list(zip(img_info, img_info[1:]))
        massages = []
        for item in marge_split_list:
            image1, image2 = item

            message = self.llm_model.set_diff_prompt_image_message(
                prompt_image_list=[("Image 1", [image1.image_path]), ("Image 2", [image2.image_path])],
                system_info_type=2
            )
            massages.append(message)

        results = await self.llm_model.predict(messages=massages, schema=Description)
        if len(results) == 0:
            return markdown_file_result

        for index, item in enumerate(results):
            if not isinstance(item, ModelInfo):
                continue

            content = item.content
            result = Description.model_validate_json(content)

            if result.truncated:
                img_info[index].merge_position = [index, index + 1]

            if result.reason:
                img_info[index].merge_reason = result.reason

        return markdown_file_result

    async def run(self, file_parsing_data: list[FileParsingResult]) -> MarkdownFileResult:
        img_info: list[ImgMergeInfo] = []
        children = []
        for page_index, file_parsing_result in enumerate(file_parsing_data):
            page_info = file_parsing_result.img_info
            img_merge_info = ImgMergeInfo(
                page_index=page_info.page_index,
                image_path=page_info.image_path,
                width=page_info.width,
                height=page_info.height,
            )
            img_info.append(img_merge_info)
            children.append(self.generate_blocks(file_parsing_result.blocks))

        markdown_file_result = MarkdownFileResult(
            ignore_block_label=list(dict.fromkeys(self.ignore_labels + self.ignore_footer_label)),
            img_info=img_info,
            children=children
        )

        if self.is_merge:
            markdown_file_result = await self.marge_paper(markdown_file_result)

        return markdown_file_result
