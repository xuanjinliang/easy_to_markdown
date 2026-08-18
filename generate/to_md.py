from pkg.common import ensure_dir
import os
from pathlib import Path
from uuid import uuid4
import shutil

from generate import (FileParsingResult, ParsingResult, MarkdownInfo, MarkdownFileResult,
                      ModelInfo, TableInfo)


class MarkdownWriter:
    def __init__(self, file_path: str, mode: str = "w", ignore_labels: list[str] | None = None):
        self.ignore_labels = [] if ignore_labels is None else ignore_labels
        self.md_file = open(file_path, mode, encoding="utf-8")

    def write(self, content: str):
        self.md_file.write(content)
        self.md_file.flush()

    def write_line(self, content: str):
        self.write(content + "\n")

    def close(self):
        if not self.md_file.closed:
            self.md_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class MarkdownJsonWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        ensure_dir(output_dir)

    def set_content(self, block: ParsingResult) -> MarkdownInfo:
        markdown_info = MarkdownInfo(
            block_id=block.block_id,
            block_label=block.block_label,
            block_bbox=block.block_bbox,
            block_label_type=block.block_label_type,
            level=block.level,
            block_content="",
            image_path=None
        )

        label_type = block.block_label_type

        match label_type:
            case "title":
                if isinstance(block.block_content, ModelInfo):
                    markdown_info.block_content = f"{'#' * block.level} {block.block_content.content}\n\n"
            case "image":
                if block.crop_path is not None and os.path.exists(block.crop_path):
                    markdown_info.image_path = self.set_image_content(block.crop_path)
            case "table":
                markdown_info.block_content = ""
            case _:
                if isinstance(block.block_content, ModelInfo):
                    markdown_info.block_content = f"{block.block_content.content}\n"

        return markdown_info

    def set_table_content(self, table_info: TableInfo):
        pass

    def set_image_content(self, image_path: str) -> str:
        original = Path(image_path)
        dst_dir = os.path.join(self.output_dir, "images")
        ensure_dir(dst_dir)

        dst = Path(dst_dir) / f"{uuid4()}{original.suffix}"

        shutil.copy2(original, dst)

        image_path = str(dst.relative_to(self.output_dir))

        return f'<img src="{image_path}" alt="Image" />'

    def generate_blocks(self, blocks: list[ParsingResult]) -> list[MarkdownInfo]:

        blocks_info_list: list[MarkdownInfo] = []
        for block in blocks:
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
