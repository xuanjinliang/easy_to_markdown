from typing import Literal
from generate import ParsingResult
from pkg.label_normalization import map_paddle_label
from pkg.enum_class import BlockType
from pkg.coordinate_overlap import get_overlap_result_np


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
        threshold=0.6,
        mode: Literal["max", "min"] = "max"
):
    if len(blocks) < 2:
        return

    keep_block = blocks[0]
    for block in blocks[1:]:
        if block.block_label_type != keep_block.block_label_type:
            continue

        ratio, are1, are2 = calc_overlap_ratio(keep_block, block)
        if ratio < threshold:
            continue

        if mode == "max":
            if are1 > are2:
                block.remove = True
            else:
                keep_block.remove = True
                keep_block = block
        else:
            if are1 < are2:
                block.remove = True
            else:
                keep_block.remove = True
                keep_block = block


def remove_repeat_blocks(blocks: list[ParsingResult]) -> list[ParsingResult]:
    block_bbox_list = [parsing_result.block_bbox for parsing_result in blocks]
    overlap_result = get_overlap_result_np(block_bbox_list, no_duplicate=True)
    for item in overlap_result:
        if len(item.overlap) == 0:
            continue
        filter_overlap_by_area([blocks[item.index]] + [blocks[index] for index in item.overlap])

    return blocks
