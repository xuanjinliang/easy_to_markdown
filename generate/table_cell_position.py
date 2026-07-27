import os
from pathlib import Path
from pkg.common import ensure_dir
from pkg.label_normalization import map_paddle_label
from pkg.draw_label import BboxLabel, DrawImageLabel, draw_labels
from generate import CellInfo, TableInfo, RowInfo, ColumnsInfo, ParsingResult, ImageResponse
from typing import Any, Literal
import numpy as np
import cv2
from PIL import Image
from itertools import chain

import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


def normalize_bbox_np(coords: np.ndarray) -> np.ndarray:
    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 2]
    y2 = coords[:, 3]

    left = np.minimum(x1, x2)
    top = np.minimum(y1, y2)
    right = np.maximum(x1, x2)
    bottom = np.maximum(y1, y2)

    return np.stack((left, top, right, bottom), axis=1).astype(np.float32)


def nms_numpy(
        bboxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.8
) -> np.ndarray:
    if bboxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        inter_x1 = np.maximum(x1[i], x1[rest])
        inter_y1 = np.maximum(y1[i], y1[rest])
        inter_x2 = np.minimum(x2[i], x2[rest])
        inter_y2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union = areas[i] + areas[rest] - inter_area

        iou = np.zeros_like(union, dtype=np.float32)
        valid = union > 0.0
        iou[valid] = inter_area[valid] / union[valid]

        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def cluster_1d(values: np.ndarray, tol: float = 5) -> np.ndarray:
    values = np.sort(values)

    if len(values) == 0:
        return np.array([], dtype=np.float32)

    clusters = []
    current = [values[0]]

    for v in values[1:]:
        if abs(v - current[-1]) <= tol:
            current.append(v)
        else:
            clusters.append(np.mean(current))
            current = [v]

    clusters.append(np.mean(current))

    return np.asarray(clusters, dtype=np.float32)


def merge_intervals_np(
        intervals: np.ndarray, tol: float = 5) -> np.ndarray:
    if len(intervals) == 0:
        return np.empty((0, 2), dtype=np.float32)

    intervals = intervals[np.argsort(intervals[:, 0])]

    merged = []
    cur_start, cur_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= cur_end + tol:
            cur_end = float(max(cur_end, end))
        else:
            merged.append([cur_start, cur_end])
            cur_start, cur_end = start, end

    merged.append([cur_start, cur_end])

    return np.asarray(merged, dtype=np.float32)


def is_full_width_np(
        intervals: np.ndarray,
        min_x: float,
        max_x: float, tol: float = 5) -> bool:
    if len(intervals) == 0:
        return False

    merged = merge_intervals_np(intervals, tol=tol)

    if len(merged) == 0:
        return False

    left_ok = merged[0, 0] <= min_x + tol
    right_ok = merged[-1, 1] >= max_x - tol

    return bool(left_ok and right_ok)


def is_full_height_np(
        intervals: np.ndarray,
        min_y: float,
        max_y: float, tol: float = 5) -> bool:
    if len(intervals) == 0:
        return False

    merged = merge_intervals_np(intervals, tol=tol)

    if len(merged) == 0:
        return False

    top_ok = merged[0, 0] <= min_y + tol
    bottom_ok = merged[-1, 1] >= max_y - tol

    return bool(top_ok and bottom_ok)


def group_cells_by_columns_np(
        row_list: list[list[CellInfo]],
        tol: float = 5,
        min_overlap_ratio=0.05
) -> list[list[list[CellInfo]]]:
    if not row_list:
        return []

    table_list: list[list[list[CellInfo]]] = []
    for row in row_list:
        coords = np.asarray([c.bbox for c in row], dtype=np.float32)
        x1 = coords[:, 0]
        y1 = coords[:, 1]
        x2 = coords[:, 2]
        y2 = coords[:, 3]

        min_x = np.min(x1)
        max_x = np.max(x2)
        min_y = np.min(y1)
        max_y = np.max(y2)

        # 1. 聚类所有 x 边界
        all_x = np.concatenate([x1, x2])
        x_lines = cluster_1d(all_x, tol=tol)

        # 2. 找真实行分割线
        column_lines = [min_x]

        for x in x_lines:
            # 跳过表格最左和最右边界
            if abs(x - min_x) <= tol or abs(x - max_x) <= tol:
                continue

            # 找出左边界或右边界接近该 x 的 cell
            mask = (np.abs(x1 - x) <= tol) | (np.abs(x2 - x) <= tol)

            intervals = coords[mask][:, [1, 3]]

            if is_full_height_np(intervals, min_y, max_y, tol=tol):
                column_lines.append(x)

        column_lines.append(max_x)

        column_lines = np.asarray(column_lines, dtype=np.float32)
        column_lines = np.sort(cluster_1d(column_lines, tol=tol))

        # 3. 根据真实行区间归类 cell
        row_list: list[list[CellInfo]] = []

        cell_widths = x2 - x1

        for i in range(len(column_lines) - 1):
            row_left = column_lines[i]
            row_right = column_lines[i + 1]
            row_width = row_right - row_left

            overlap = np.minimum(x2, row_right) - np.maximum(x1, row_left)
            overlap = np.maximum(overlap, 0)

            # 重叠比例，可以相对 cell 宽度，也可以相对列宽度
            ratio_by_cell = overlap / np.maximum(cell_widths, 1)
            ratio_by_column = overlap / max(row_width, 1)

            mask = (overlap > tol) | (ratio_by_cell >= min_overlap_ratio) | (ratio_by_column >= min_overlap_ratio)

            column_indices = np.where(mask)[0]

            # 每行内部按 y1 排序，再按 x1 排序
            if len(column_indices) > 0:
                column_indices = column_indices[np.lexsort((x1[column_indices], y1[column_indices]))]

            column_cells = [row[int(idx)] for idx in column_indices]
            row_list.append(column_cells)

        table_list.append(row_list)

    return table_list


def group_cells_by_rows_np(
        cells: list[CellInfo],
        tol: float = 5,
        min_overlap_ratio=0.05) -> list[list[CellInfo]]:
    if not cells:
        return []

    coords = np.asarray([c.bbox for c in cells], dtype=np.float32)

    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 2]
    y2 = coords[:, 3]

    min_x = np.min(x1)
    max_x = np.max(x2)
    min_y = np.min(y1)
    max_y = np.max(y2)

    # 1. 聚类所有 y 边界
    all_y = np.concatenate([y1, y2])
    y_lines = cluster_1d(all_y, tol=tol)

    # 2. 找真实行分割线
    row_lines = [min_y]

    for y in y_lines:
        # 跳过表格最上和最下边界
        if abs(y - min_y) <= tol or abs(y - max_y) <= tol:
            continue

        # 找出上边界或下边界接近该 y 的 cell
        mask = (np.abs(y1 - y) <= tol) | (np.abs(y2 - y) <= tol)

        intervals = coords[mask][:, [0, 2]]

        if is_full_width_np(intervals, min_x, max_x, tol=tol):
            row_lines.append(y)

    row_lines.append(max_y)

    row_lines = np.asarray(row_lines, dtype=np.float32)
    row_lines = np.sort(cluster_1d(row_lines, tol=tol))

    # 3. 根据真实行区间归类 cell
    rows: list[list[CellInfo]] = []

    cell_heights = y2 - y1

    for i in range(len(row_lines) - 1):
        row_top = row_lines[i]
        row_bottom = row_lines[i + 1]
        row_height = row_bottom - row_top

        overlap = np.minimum(y2, row_bottom) - np.maximum(y1, row_top)
        overlap = np.maximum(overlap, 0)

        # 重叠比例，可以相对 cell 高度，也可以相对行高度
        ratio_by_cell = overlap / np.maximum(cell_heights, 1)
        ratio_by_row = overlap / max(row_height, 1)

        mask = (overlap > tol) | (ratio_by_cell >= min_overlap_ratio) | (ratio_by_row >= min_overlap_ratio)

        row_indices = np.where(mask)[0]

        # 每行内部按 x1 排序，再按 y1 排序
        if len(row_indices) > 0:
            row_indices = row_indices[np.lexsort((y1[row_indices], x1[row_indices]))]

        row_cells = [cells[int(idx)] for idx in row_indices]
        rows.append(row_cells)

    return rows


def draw_cell_label(
        img_path: str,
        coords_list: list[list[float]],
        scores_list: list[float]
) -> str | None:
    if not img_path or not coords_list:
        return None

    bbox_label_list: list[BboxLabel] = []
    for i, item in enumerate(coords_list):
        bbox_label = BboxLabel(
            block_label=f"{i}_{scores_list[i]:.2f}",
            block_bbox=item
        )
        bbox_label_list.append(bbox_label)

    file_path = Path(img_path)
    filename = file_path.stem
    folder = file_path.parent
    output_dir = os.path.join(folder, filename, "table_crop")
    ensure_dir(output_dir)

    draw_info = DrawImageLabel(
        img_path=str(file_path),
        bbox_label=bbox_label_list,
    )

    img = draw_labels(draw_info)

    if img is None:
        return None

    img_path = os.path.join(output_dir, f"{filename}_sheet.webp")
    if not cv2.imwrite(img_path, img):
        logger.error("draw cell container failed")
        return None

    return img_path


def format_table_info(
        table_info: TableInfo,
        col_info: list[list[list[CellInfo]]]) -> TableInfo:
    output_dir = table_info.img_output_dir

    img = Image.open(table_info.image_path).convert("RGB")
    img_w, img_h = img.size

    table_list: list[RowInfo] = []
    for i, rows in enumerate(col_info):
        if not rows:
            continue

        rows_list: list[ColumnsInfo] = []

        for j, col in enumerate(rows):
            if not col:
                continue

            coords = np.asarray([c.bbox for c in col], dtype=np.float32)

            x1 = coords[:, 0]
            y1 = coords[:, 1]
            x2 = coords[:, 2]
            y2 = coords[:, 3]

            min_x = np.min(x1)
            max_x = np.max(x2)
            min_y = np.min(y1)
            max_y = np.max(y2)

            crop_x1 = max(0, int(min_x))
            crop_y1 = max(0, int(min_y))
            crop_x2 = min(img_w, int(max_x))
            crop_y2 = min(img_h, int(max_y))

            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            crop_name = f'row_{i}_{j}.webp'
            crop_path = os.path.join(output_dir, crop_name)

            crop_img.save(crop_path)

            columns_info = ColumnsInfo(
                image_path=crop_path,
                width=float(max_x) - float(min_x),
                height=float(max_y) - float(min_y),
                bbox=[float(min_x), float(min_y), float(max_x), float(max_y)],
                columns_list=col,
            )
            rows_list.append(columns_info)

        if not rows_list:
            continue

        all_col = list(chain.from_iterable(rows))
        coords = np.asarray([c.bbox for c in all_col], dtype=np.float32)
        x1 = coords[:, 0]
        y1 = coords[:, 1]
        x2 = coords[:, 2]
        y2 = coords[:, 3]

        min_x = np.min(x1)
        max_x = np.max(x2)
        min_y = np.min(y1)
        max_y = np.max(y2)

        row_info = RowInfo(
            width=float(max_x) - float(min_x),
            height=float(max_y) - float(min_y),
            bbox=[float(min_x), float(min_y), float(max_x), float(max_y)],
            rows_list=rows_list
        )

        table_list.append(row_info)

    table_info.table_list = table_list
    return table_info


def clean_cell_detections(
        img_path: str,
        detections: list[dict[str, Any]],
        label_name: str = "cell",
        score_threshold: float = 0.25,
        use_nms: bool = True,
        nms_iou_threshold: float = 0.8,
        border_diff_threshold: float = 5
) -> TableInfo | None:
    if not detections or not img_path:
        return None

    coords_list: list[list[float]] = []
    scores_list: list[float] = []
    cls_ids_list: list[Any] = []

    for item in detections:
        if item.get("label") != label_name:
            continue

        score = float(item.get("score", 0.0))
        if score < score_threshold:
            continue

        coord = item.get("coordinate", [])
        if coord is None or len(coord) != 4:
            continue

        coords_list.append(coord)
        scores_list.append(score)
        cls_ids_list.append(item.get("cls_id", 0))

    if not coords_list:
        return None

    draw_img_path = draw_cell_label(
        img_path=img_path,
        coords_list=coords_list,
        scores_list=scores_list
    )

    if draw_img_path is None:
        return None

    output_dir = Path(draw_img_path).parent

    coords = np.asarray(coords_list, dtype=np.float32)
    scores = np.asarray(scores_list, dtype=np.float32)

    min_x1, min_y1 = coords[:, :2].min(axis=0)
    max_x2, max_y2 = coords[:, 2:].max(axis=0)
    table_width = max_x2 - min_x1
    table_height = max_y2 - min_y1

    table_info = TableInfo(
        draw_img_path=draw_img_path,
        image_path=img_path,
        width=table_width,
        height=table_height,
        bbox=[min_x1, min_y1, max_x2, max_y2],
        img_output_dir=str(output_dir)
    )

    min_width = min(table_width * 0.01, 5)
    min_height = min(table_height * 0.01, 5)
    min_area = min((table_width * 0.01) * (table_height * 0.01), 30)

    bboxes = normalize_bbox_np(coords)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    widths = x2 - x1
    heights = y2 - y1
    areas = widths * heights

    valid_mask = (
            (widths > 0.0) &
            (heights > 0.0) &
            (widths >= min_width) &
            (heights >= min_height) &
            (areas >= min_area)
    )

    if not np.any(valid_mask):
        return table_info

    bboxes = bboxes[valid_mask]
    scores = scores[valid_mask]
    widths = widths[valid_mask]
    heights = heights[valid_mask]
    areas = areas[valid_mask]

    cls_ids_array = np.asarray(cls_ids_list, dtype=np.int64)[valid_mask]

    if use_nms and bboxes.shape[0] > 1:
        keep_indices = nms_numpy(
            bboxes=bboxes,
            scores=scores,
            iou_threshold=nms_iou_threshold
        )

        bboxes = bboxes[keep_indices]
        scores = scores[keep_indices]
        widths = widths[keep_indices]
        heights = heights[keep_indices]
        areas = areas[keep_indices]
        cls_ids_array = cls_ids_array[keep_indices]

    cells: list[CellInfo] = []

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        left = float(bbox[0])
        top = float(bbox[1])
        right = float(bbox[2])
        bottom = float(bbox[3])

        width = float(widths[i])
        height = float(heights[i])
        area = float(areas[i])

        cells.append(CellInfo(
            uid=i,
            cls_id=cls_ids_array[i],
            label=label_name,
            score=float(scores[i]),
            bbox=[left, top, right, bottom],
            width=width,
            height=height,
            cx=(left + right) * 0.5,
            cy=(top + bottom) * 0.5,
            area=area
        ))

    sort_row_list = group_cells_by_rows_np(
        cells,
        tol=border_diff_threshold
    )

    col_info = group_cells_by_columns_np(
        sort_row_list,
        tol=border_diff_threshold
    )

    return format_table_info(
        table_info=table_info,
        col_info=col_info
    )


# 过滤过小的表格cell
def table_cell_category(
        blocks: list[ParsingResult],
        cell_image: ImageResponse,
        table_w: float,
        table_h: float) -> Literal["formula", "text", "unknown"]:
    limit_w = max(table_w * 0.1, 50)
    limit_h = max(table_h * 0.1, 50)

    if cell_image.width < limit_w or cell_image.height < limit_h:
        return "text"

    blocks_label_list = []
    for block in blocks:
        if block.remove:
            continue

        block_label = block.block_label
        canonical = map_paddle_label(block_label)
        block.block_label_type = canonical

        x1, y1, x2, y2 = block.block_bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        if w < limit_w or h < limit_h:
            continue

        match canonical:
            case "table" | "image":
                return "unknown"
            case "formula":
                blocks_label_list.append("formula")
            case _:
                blocks_label_list.append("text")

    if "formula" in blocks_label_list:
        return "formula"

    return "text"
