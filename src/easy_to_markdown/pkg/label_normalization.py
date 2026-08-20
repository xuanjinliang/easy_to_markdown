from easy_to_markdown.pkg.enum_class import BlockType


def map_paddle_label(label: str) -> str:
    match label:
        case BlockType.PARAGRAPH_TITLE | BlockType.DOC_TITLE:
            return "title"
        case BlockType.IMAGE | BlockType.CHART | BlockType.FOOTER_IMAGE | BlockType.HEADER_IMAGE | BlockType.SEAL:
            return "image"
        case BlockType.TABLE:
            return "table"
        case BlockType.FIGURE_TITLE:
            return "caption"
        case BlockType.VISION_FOOTNOTE:
            return "caption"
        case BlockType.INLINE_FORMULA | BlockType.DISPLAY_FORMULA:
            return "formula"
        case _:
            return "text"
