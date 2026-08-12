import cv2
import math
from typing import Tuple
from pathlib import Path
from pydantic import BaseModel, Field

font_face = cv2.FONT_HERSHEY_SIMPLEX


class BboxLabel(BaseModel):
    block_label: str
    block_bbox: list[float] = Field(..., min_length=4, max_length=4)


class DrawImageLabel(BaseModel):
    img_path: str
    img_width: float = 0
    img_height: float = 0
    bbox_label: list[BboxLabel] = []
    font_scale: float = 1
    font_space: int = 4
    border_line: int = 5
    border_space: int = 4
    font_pos_step: int = 10
    thickness: int = 2


def overlap(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return not (
            ax2 <= bx1 or
            bx2 <= ax1 or
            ay2 <= by1 or
            by2 <= ay1
    )


def distance(a, b):
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)


def get_text_size(
        text: str,
        font_scale: float,
        thickness: int,
        font_space: int
) -> Tuple[float, float]:
    (w, h), _ = cv2.getTextSize(
        text,
        font_face,
        font_scale,
        thickness
    )
    return w + font_space, h + font_space


def generate_candidates(
        xyxy: Tuple[int, int, int, int],
        page_w: float, page_h: float,
        tw: float, th: float,
        border_space: int, step=10
) -> list[Tuple[float, float, float, float]]:
    candidates = []

    x1, y1, x2, y2 = xyxy

    # up
    y = y1 - th - border_space
    for x in range(x1, x2, step):
        if y < 0:
            continue

        if x + tw >= page_w or x < 0:
            continue
        candidates.append((x, y, x + tw, y + th))

    # right
    x = x2 + border_space
    for y in range(int(y1 - th), y2, step):
        if x + tw >= page_w:
            continue

        if y + th >= page_h or y < 0:
            continue

        candidates.append((x, y, x + tw, y + th))

    # down
    y = y2 + border_space
    for x in range(int(x1 - tw), x2, step):
        if y + th > page_h:
            continue

        if x < 0 or x + tw >= page_w:
            continue
        candidates.append((x, y, x + tw, y + th))

    # left
    x = x1 - tw - border_space
    for y in range(int(y1 - th), y2, step):
        if x < 0:
            continue

        if y + th >= page_h or y < 0:
            continue
        candidates.append((x, y, x + tw, y + th))

    return candidates


def is_free(box: Tuple[float, float, float, float], occupied: list[Tuple[float, float, float, float]]) -> bool:
    if len(occupied) == 0:
        return False

    for o in occupied:
        if not overlap(box, o):
            return False

    return True


def draw_labels(draw_info: DrawImageLabel) -> cv2.typing.MatLike | None:
    if len(draw_info.bbox_label) == 0:
        return None

    page_w = draw_info.img_width
    page_h = draw_info.img_height
    image_path = Path(draw_info.img_path)
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"can't read the image：{str(image_path)}")

    if page_w <= 0 or page_h <= 0:
        page_h, page_w = img.shape[:2]

    colors = [
        (0, 0, 255),  # 红   RGB(255,0,0) -> BGR
        (0, 255, 0),  # 绿   RGB(0,255,0)
        (255, 0, 0),  # 蓝   RGB(0,0,255)
        (0, 165, 255),  # 橙   RGB(255,165,0)
        (128, 0, 128),  # 紫   RGB(128,0,128)
        (153, 102, 0),  # 深蓝紫青 RGB(0,102,153)
        (255, 0, 255),  # 品红 RGB(255,0,255)
        (130, 0, 75),  # 靛蓝 RGB(75,0,130)
        (255, 128, 0),  # 天蓝 RGB(0,128,255)
        (255, 20, 147)  # 深粉红 RGB(147,20,255)
    ]

    len_colors = len(colors)

    occupied = []
    results = []

    for i, block in enumerate(draw_info.bbox_label):
        x1, y1, x2, y2 = map(int, block.block_bbox)
        if x2 <= x1 or y2 <= y1:
            continue

        text = block.block_label
        base_color = colors[i % len_colors]

        # print(f"base_color: {base_color}, index: {i}, text: {text}")
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            base_color,
            draw_info.border_line
        )

        tw, th = get_text_size(
            text=text,
            font_scale=draw_info.font_scale,
            thickness=draw_info.thickness,
            font_space=draw_info.font_space
        )

        candidates = generate_candidates(
            (x1, y1, x2, y2),
            page_w, page_h,
            tw, th, draw_info.border_space, draw_info.font_pos_step)

        draw_position = None
        for c in candidates:
            if is_free(c, occupied):
                continue

            draw_position = c
            break

        if draw_position is not None:
            occupied.append(draw_position)
            results.append((text, draw_position, base_color))
        else:
            results.append((text, (x1, y1, x2, y2), base_color))

    for item in results:
        text, draw_position, base_color = item
        ax1, ay1, ax2, ay2 = draw_position
        cv2.putText(
            img,
            text,
            (int(ax1), int(ay2)),
            font_face,
            draw_info.font_scale,
            base_color,
            draw_info.thickness
        )

    return img
