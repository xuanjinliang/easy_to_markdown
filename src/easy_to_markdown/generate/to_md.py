from easy_to_markdown.pkg.common import ensure_dir
import os
from pathlib import Path
from uuid import uuid4
import shutil
from easy_to_markdown.pkg.enum_class import BlockType
from easy_to_markdown.pkg.format_table import Table, TableCell
from easy_to_markdown.generate import (FileParsingResult, ParsingResult, MarkdownInfo, MarkdownFileResult,
                                       ModelInfo, TableInfo)


class MarkdownWriter:
    def __init__(self, file_path: str, mode: str = "w",
                 ignore_labels: list[str] | None = None,
                 ignore_header: bool = True,
                 ignore_footer: bool = True):

        ignore_labels = [] if ignore_labels is None else ignore_labels
        if ignore_header:
            ignore_labels += [BlockType.HEADER, BlockType.HEADER_IMAGE]

        self.ignore_footer_label = [BlockType.FOOTER, BlockType.FOOTER_IMAGE,
                                    BlockType.FOOTNOTE] if ignore_footer else []

        self.ignore_labels = list(dict.fromkeys(ignore_labels))
        self.md_file = open(file_path, mode, encoding="utf-8")

    def write(self, markdown_info: MarkdownInfo):
        if len(self.ignore_labels) > 0 and markdown_info.block_label in self.ignore_labels:
            return

        content = markdown_info.block_image_content if (
                markdown_info.block_image_content is not None) else markdown_info.block_content
        self.md_file.write(content)
        self.md_file.flush()

    def write_list(self, list_markdown_info: list[MarkdownInfo]):
        if len(list_markdown_info) == 0:
            return

        for markdown_info in list_markdown_info:
            if len(self.ignore_footer_label) > 0 and markdown_info.block_label in self.ignore_footer_label:
                break
            self.write(markdown_info)

    def close(self):
        if not self.md_file.closed:
            self.md_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class MarkdownJsonWriter:
    def __init__(self, output_dir: str, tolerance: float = 5.0):
        self.output_dir = output_dir
        ensure_dir(output_dir)

        self.tolerance = tolerance

    def set_content(self, block: ParsingResult) -> MarkdownInfo:
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

    def set_table_content(self, table_info: TableInfo) -> str:
        if len(table_info.table_list) == 0:
            return ""

        table = Table(width=table_info.width, height=table_info.height)

        for rows in table_info.table_list:
            table_cell = []
            for cell in rows.rows_list:
                if cell.columns_blocks is not None:
                    blocks = cell.columns_blocks.blocks
                    content = ""
                    for block in blocks:
                        markdown_info = self.set_content(block)
                        content += markdown_info.block_content
                    table_cell.append(TableCell(html=content, bbox=cell.bbox))
                    continue

                content = cell.block_content.content if cell.block_content is not None else ""
                table_cell.append(TableCell(text=content, bbox=cell.bbox))

            if len(table_cell) > 0:
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
            if block.remove:
                continue

            md_info = self.set_content(block)
            if len(md_info.block_content) == 0 and md_info.block_image_content is None:
                continue

            blocks_info_list.append(self.set_content(block))

        return blocks_info_list

    def run(self, file_parsing_data: list[FileParsingResult]) -> list[MarkdownFileResult]:
        markdown_file_list: list[MarkdownFileResult] = []
        for page_index, file_parsing_result in enumerate(file_parsing_data):
            markdown_file_info = MarkdownFileResult(
                img_info=file_parsing_result.img_info,
                page=(page_index + 1),
                children=self.generate_blocks(file_parsing_result.blocks)
            )

            markdown_file_list.append(markdown_file_info)

        return markdown_file_list
