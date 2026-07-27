import unittest
from generate import ParsingResult
from pkg.draw_label import generate_candidates, get_text_size
from pkg.pdf_to_image import ImageResponse


class TestFontPosition(unittest.TestCase):
    def test_place_labels(self):
        image_response = ImageResponse(
            page_index=1,
            image_path="/Users/xuanjinliang/PycharmProjects/pdf_local_model/pdf_temp/aws_2023_5fe9b746-79c3-4fc6-a996-b48ada7dcbeb/pdf_image/page_1.webp",
            width=1191,
            height=1685
        )
        block = ParsingResult(
            block_id=5,
            block_label="abandon",
            block_bbox=[
                30.79,
                17.28,
                550.31,
                39.32
            ],
            block_bbox_norm=[
                0.025856,
                0.010258,
                0.462057,
                0.023338
            ],
            group_id=5,
            score=0.7147,
        )

        text = f'{block.block_id}_{block.block_label}_{block.score:.2f}'

        tw, th = get_text_size(
            text=text, font_scale=0.6, thickness=1, font_space=4)

        print(f"tw --> {tw}, th --> {th}")
        x1, y1, x2, y2 = map(int, block.block_bbox)
        candidates = generate_candidates(
            xyxy=(x1, y1, x2, y2),
            page_w=image_response.width,
            page_h=image_response.height,
            tw=tw,
            th=th,
            border_space=4
        )
        print(f"candidates --> {candidates}")
