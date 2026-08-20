from typing import Literal
from easy_to_markdown.generate import ParsingResult
from easy_to_markdown.pkg.label_normalization import map_paddle_label
from easy_to_markdown.pkg.enum_class import BlockType
from easy_to_markdown.pkg.coordinate_overlap import get_overlap_result_np


def set_block_process(blocks: list[ParsingResult]) -> list[ParsingResult]:
    for block in blocks:
        block_label = block.block_label
        block.block_label_type = map_paddle_label(block_label)

        match block_label:
            case BlockType.DOC_TITLE:
                block.level = 1
            case BlockType.PARAGRAPH_TITLE:
                block.level = 2
            case _:
                block.level = 0

    return blocks


def calc_overlap_ratio(block1: ParsingResult, block2: ParsingResult) -> tuple[float, float, float]:
    box1 = block1.block_bbox
    box2 = block2.block_bbox

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0

    intersection = (
            (x2 - x1) *
            (y2 - y1)
    )

    area1 = (
            (box1[2] - box1[0]) *
            (box1[3] - box1[1])
    )

    area2 = (
            (box2[2] - box2[0]) *
            (box2[3] - box2[1])
    )

    return intersection / min(area1, area2), area1, area2


def filter_overlap_by_area(
        blocks: list[ParsingResult],
        threshold,
        mode: Literal["max", "min"]
):
    if len(blocks) < 2:
        return

    keep_block = None
    for block in blocks:
        if block.remove:
            continue

        if keep_block is None:
            keep_block = block
            continue

        if block.block_label_type == "image" or keep_block.block_label_type == "image":
            continue

        ratio, are1, are2 = calc_overlap_ratio(keep_block, block)
        if ratio < threshold:
            continue

        if mode == "max":
            bbox, bbox_norm = get_max_bbox(blocks=[keep_block, block])

            if are1 > are2:
                block.remove = True
            else:
                keep_block.remove = True
                keep_block = block

            keep_block.block_bbox = bbox
            keep_block.block_bbox_norm = bbox_norm
        else:
            if are1 < are2:
                block.remove = True
            else:
                keep_block.remove = True
                keep_block = block


def remove_repeat_blocks(
        blocks: list[ParsingResult],
        threshold=0.5,
        mode: Literal["max", "min"] = "max"
) -> list[ParsingResult]:
    block_bbox_list = [parsing_result.block_bbox for parsing_result in blocks]
    overlap_result = get_overlap_result_np(block_bbox_list, no_duplicate=True)
    for item in overlap_result:
        if len(item.overlap) == 0:
            continue
        filter_overlap_by_area(
            [blocks[item.index]] + [blocks[index] for index in item.overlap],
            threshold=threshold,
            mode=mode
        )

    return blocks


def get_max_bbox(blocks: list[ParsingResult]) -> tuple[list[float], list[float]]:
    x1, y1, x2, y2 = blocks[0].block_bbox
    x_1, y_1, x_2, y_2 = blocks[1].block_bbox

    x1_norm, y1_norm, x2_norm, y2_norm = blocks[0].block_bbox_norm
    x_1_norm, y_1_norm, x_2_norm, y_2_norm = blocks[1].block_bbox_norm

    return ([min(x1, x_1), min(y1, y_1), max(x2, x_2), max(y2, y_2)],
            [min(x1_norm, x_1_norm), min(y1_norm, y_1), max(x2_norm, x_2_norm), max(y2_norm, y_2_norm)])
