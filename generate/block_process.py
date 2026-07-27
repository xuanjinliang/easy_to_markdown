from generate import ParsingResult
from pkg.label_normalization import map_paddle_label
from pkg.enum_class import BlockType


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
