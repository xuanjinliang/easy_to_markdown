import cv2
from easy_to_markdown.generate import ParsingResult
from easy_to_markdown.pkg.pdf_to_image import ImageResponse
from src.easy_to_markdown.pkg.draw_label import draw_labels, BboxLabel, DrawImageLabel


def layout_labels(
        image_info: ImageResponse,
        blocks: list[ParsingResult],
        font_scale: float = 1,
        font_space: int = 4,
        border_line: int = 5,
        border_space: int = 4,
        font_pos_step: int = 10,
        thickness: int = 2
) -> cv2.typing.MatLike | None:
    bbox_label_list: list[BboxLabel] = []
    for i, block in enumerate(blocks):
        if block.remove:
            continue

        block_id = block.block_id
        label = block.block_label
        score = block.score

        text = f'{block_id}_{label}_{score:.2f}'

        bbox_label_list.append(BboxLabel(
            block_label=text,
            block_bbox=block.block_bbox
        ))

    return draw_labels_info(
        image_info=image_info,
        bbox_label_list=bbox_label_list,
        font_scale=font_scale,
        font_space=font_space,
        border_line=border_line,
        border_space=border_space,
        font_pos_step=font_pos_step,
        thickness=thickness
    )


def draw_labels_info(image_info: ImageResponse,
                     bbox_label_list: list[BboxLabel],
                     font_scale: float = 1,
                     font_space: int = 4,
                     border_line: int = 5,
                     border_space: int = 4,
                     font_pos_step: int = 10,
                     thickness: int = 2) -> cv2.typing.MatLike | None:
    draw_info = DrawImageLabel(
        img_path=image_info.image_path,
        img_width=image_info.width,
        img_height=image_info.height,
        bbox_label=bbox_label_list,
        font_scale=font_scale,
        font_space=font_space,
        border_line=border_line,
        border_space=border_space,
        font_pos_step=font_pos_step,
        thickness=thickness,
    )

    return draw_labels(draw_info)
