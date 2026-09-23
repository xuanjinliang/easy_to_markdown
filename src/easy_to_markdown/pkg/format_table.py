from __future__ import annotations
from lxml import etree
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal, Iterable
import numpy as np
import re
import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


class TableCell(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
    )
    text: str = ""
    row: int = Field(default=0, ge=0)
    col: int = Field(default=0, ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    tag: Literal["td", "th"] = "td"
    bbox: Optional[list[float]] = Field(default=None, min_length=4, max_length=4)
    is_original_rowspan: bool = True


class TableRow(BaseModel):
    cells: list[TableCell] = Field(default_factory=list)


class Table(BaseModel):
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    origin_x: float = 0.0
    origin_y: float = 0.0
    rows: list[TableRow] = Field(
        default_factory=list
    )
    caption: str | None = None

    def add_row(self, cells: list[TableCell]) -> TableRow:
        row = TableRow(
            cells=list(cells)
        )

        self.rows.append(row)
        return row

    def get_cells(self) -> list[TableCell]:
        return [
            cell
            for row in self.rows
            for cell in row.cells
        ]

    def _get_bbox_array(self) -> np.ndarray:
        cells = self.get_cells()
        if not cells:
            return np.empty(
                (0, 4),
                dtype=np.float64,
            )

        if any(cell.bbox is None for cell in cells):
            raise ValueError("All cells must have bbox")

        return np.asarray(
            [cell.bbox for cell in cells],
            dtype=np.float64,
        )

    @staticmethod
    def _cluster_coordinates(values: np.ndarray, tolerance: float) -> np.ndarray:
        if values.size == 0:
            return values

        values = np.sort(
            values.astype(np.float64)
        )

        groups = []
        current = [values[0]]

        for value in values[1:]:

            if value - current[-1] <= tolerance:
                current.append(value)
            else:
                groups.append(current)
                current = [value]

        groups.append(current)

        return np.asarray(
            [
                np.mean(group)
                for group in groups
            ],
            dtype=np.float64,
        )

    def build_grid(self, tolerance: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        bbox = self._get_bbox_array()

        if bbox.size == 0:
            raise ValueError("Table has no cells")

        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]

        x_values = np.concatenate(
            [
                x1,
                x2,
                np.asarray(
                    [
                        self.origin_x,
                        self.origin_x + self.width,
                    ]
                ),
            ]
        )

        y_values = np.concatenate(
            [
                y1,
                y2,
                np.asarray(
                    [
                        self.origin_y,
                        self.origin_y + self.height,
                    ]
                ),
            ]
        )

        x_grid = self._cluster_coordinates(
            x_values,
            tolerance,
        )

        y_grid = self._cluster_coordinates(
            y_values,
            tolerance,
        )

        return x_grid, y_grid

    @staticmethod
    def _nearest_grid_index(values: np.ndarray, grid: np.ndarray, tolerance: float) -> np.ndarray:
        distance = np.abs(
            values[:, None] - grid[None, :]
        )

        indices = np.argmin(
            distance,
            axis=1,
        )

        min_distance = distance[
            np.arange(len(values)),
            indices,
        ]

        if np.any(
                min_distance > tolerance
        ):
            raise ValueError(
                "Some bbox coordinates cannot "
                "be mapped to table grid"
            )

        return indices

    def calculate_spans(
            self,
            tolerance: float = 2.0,
    ) -> None:
        bbox = self._get_bbox_array()

        if bbox.size == 0:
            return

        x_grid, y_grid = self.build_grid(
            tolerance=tolerance
        )

        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]

        col_start = self._nearest_grid_index(
            x1,
            x_grid,
            tolerance,
        )

        col_end = self._nearest_grid_index(
            x2,
            x_grid,
            tolerance,
        )

        row_start = self._nearest_grid_index(
            y1,
            y_grid,
            tolerance,
        )

        row_end = self._nearest_grid_index(
            y2,
            y_grid,
            tolerance,
        )

        cols = col_end - col_start
        rows = row_end - row_start

        if np.any(cols <= 0):
            raise ValueError(
                "Invalid colspan detected"
            )

        if np.any(rows <= 0):
            raise ValueError(
                "Invalid rowspan detected"
            )

        cells = self.get_cells()

        for i, cell in enumerate(cells):
            cell.col = int(
                col_start[i]
            ) + 1

            cell.row = int(
                row_start[i]
            ) + 1

            cell.colspan = int(
                cols[i]
            )

            cell.rowspan = int(
                rows[i]
            )

        for row in self.rows:
            if not row.cells:
                continue

            rowspans = {cell.rowspan for cell in row.cells}

            if len(rowspans) == 1:
                for cell in row.cells:
                    cell.rowspan = 1

    def to_html(self) -> str:
        return HtmlTableRenderer().render(self)


class HtmlTableRenderer:
    border_style = "border: 1px solid black;"

    def render(self, table: Table) -> str:
        table_element = etree.Element(
            "table",
            style=f"border-collapse: collapse;{self.border_style}"
        )

        if table.caption:
            caption = etree.SubElement(
                table_element,
                "caption",
            )
            caption.text = table.caption

        # rows
        for row in table.rows:
            tr = etree.SubElement(
                table_element,
                "tr",
                style=f"{self.border_style}"
            )

            for cell in row.cells:
                self._render_cell(tr, cell)

        return etree.tostring(
            table_element,
            encoding="unicode",
            pretty_print=True,
        )

    @staticmethod
    def set_text_with_br(
            element: etree._Element,
            text: str,
    ) -> None:
        parts = text.split("\n")

        if not parts:
            return

        element.text = parts[0]
        for part in parts[1:]:
            br = etree.SubElement(
                element,
                "br",
            )
            br.tail = part

    def _render_cell(self, tr: etree._Element, cell: TableCell) -> None:

        td = etree.SubElement(tr, cell.tag, style=f"{self.border_style}")
        # rowspan
        if cell.rowspan > 1:
            td.set("rowspan", str(cell.rowspan))

        # colspan
        if cell.colspan > 1:
            td.set("colspan", str(cell.colspan), )

        self.set_text_with_br(td, cell.text)


def _get_first_row(table: Table) -> list[TableCell]:
    if len(table.rows) == 0:
        return []

    return table.rows[0].cells


def _normalize_cell_text(text: str | None) -> str:
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.casefold()

    return text


def _expand(row: list[TableCell]) -> dict[int, TableCell]:
    result: dict[int, TableCell] = {}

    for cell in row:
        start = int(cell.col)
        end = start + max(int(cell.colspan), 1)

        for col in range(start, end):
            result[col] = cell

    return result


def _bbox_center_distance(cell: TableCell, x: float) -> float:
    if cell.bbox is None:
        return 0

    x1, _, x2, _ = cell.bbox

    center = (x1 + x2) * 0.5

    return abs(center - x)


def _find_base_column_by_bbox(
        base: dict[int, TableCell], matches: dict[int, int], source_cell: TableCell
) -> int | None:
    if source_cell.bbox is None:
        return None

    source_x1, _, source_x2, _ = source_cell.bbox

    candidates: list[tuple[float, int]] = []

    for base_col, source_col in matches.items():
        base_cell = base[base_col]
        if base_cell.bbox is None:
            continue

        base_x1, _, base_x2, _ = base_cell.bbox

        overlap = min(base_x2, source_x2) - max(
            base_x1,
            source_x1,
        )

        # 有水平重叠
        if overlap > 0:
            candidates.append(
                (
                    -overlap,
                    base_col,
                )
            )

    if candidates:
        return min(candidates)[1]

    source_center = (source_x1 + source_x2) * 0.5

    return min(
        matches,
        key=lambda base_col: _bbox_center_distance(
            base[base_col],
            source_center,
        ),
    )


def compare_header(a_table: Table, b_table: Table) -> dict[int, tuple[int, ...]] | None:
    base_row = _get_first_row(a_table)
    source_row = _get_first_row(b_table)

    if not base_row or not source_row:
        return None

    base = _expand(base_row)
    source = _expand(source_row)

    base_text, base_headers, source_text, source_headers = {}, {}, {}, {}

    for item in [(base_text, base_headers, base), (source_text, source_headers, source)]:
        text_original, headers, table_row = item
        for col, cell in table_row.items():
            text = _normalize_cell_text(cell.text)
            text_original[col] = text
            if text:
                headers[col] = text

    if not base_headers:
        return None

    matches: dict[int, int] = {}
    used_source: set[int] = set()

    for base_col, base_text_value in base_headers.items():

        candidates = [
            source_col
            for source_col, source_text_value in source_headers.items()
            if (
                    source_col not in used_source
                    and source_text_value == base_text_value
            )
        ]

        if not candidates:
            return None

        source_col = min(
            candidates,
            key=lambda col: abs(col - base_col),
        )

        matches[base_col] = source_col
        used_source.add(source_col)

    source_positions = [
        matches[base_col]
        for base_col in sorted(matches)
    ]

    # compare whether the two table headers are the same
    if source_positions != sorted(source_positions):
        return None

    mapping: dict[int, list[int]] = {
        base_col: [source_col]
        for base_col, source_col in matches.items()
    }

    empty_source_cols = [
        col
        for col, text in source_text.items()
        if not text
    ]

    for source_col in empty_source_cols:
        source_cell = source[source_col]
        base_col = _find_base_column_by_bbox(
            base,
            matches,
            source_cell,
        )

        if base_col is None:
            return None

        mapping[base_col].append(source_col)

    for base_col, source_cols in mapping.items():

        source_cols.sort()

        expected = list(
            range(
                source_cols[0],
                source_cols[-1] + 1,
            )
        )

        if source_cols != expected:
            return None

    return {
        base_col: tuple(source_cols)
        for base_col, source_cols in mapping.items()
    }


def _bbox_x_overlap(a: TableCell, b: TableCell) -> float:
    if a.bbox is None or b.bbox is None:
        return 0.0

    ax1, _, ax2, _ = a.bbox
    bx1, _, bx2, _ = b.bbox

    return max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1),
    )


def _bbox_x_overlap_ratio(a: TableCell, b: TableCell) -> float:
    if a.bbox is None or b.bbox is None:
        return 0.0

    ax1, _, ax2, _ = a.bbox
    bx1, _, bx2, _ = b.bbox

    overlap = max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1),
    )

    a_width = max(ax2 - ax1, 1e-6)
    b_width = max(bx2 - bx1, 1e-6)

    return overlap / min(a_width, b_width)


def build_row_mapping(
        base_row: list[TableCell],
        source_row: list[TableCell]
) -> dict[int, tuple[int, ...]] | None:
    if not base_row or not source_row:
        return None

    base = _expand(base_row)
    source = _expand(source_row)

    if not base or not source:
        return None

    mapping: dict[int, list[int]] = {
        base_col: []
        for base_col in sorted(base)
    }

    for source_col, source_cell in source.items():

        if source_cell.bbox is None:
            return None

        candidates: list[tuple[float, float, int]] = []

        for base_col, base_cell in base.items():

            if base_cell.bbox is None:
                continue

            overlap = _bbox_x_overlap(
                base_cell,
                source_cell,
            )

            ratio = _bbox_x_overlap_ratio(
                base_cell,
                source_cell,
            )

            if overlap > 0:
                candidates.append(
                    (
                        ratio,
                        overlap,
                        base_col,
                    )
                )

        if not candidates:
            source_x1, _, source_x2, _ = source_cell.bbox
            source_center = (source_x1 + source_x2) * 0.5

            base_col = min(
                base,
                key=lambda col: _bbox_center_distance(
                    base[col],
                    source_center,
                ),
            )

        else:
            _, _, base_col = max(
                candidates,
                key=lambda item: (
                    item[0],
                    item[1],
                ),
            )

        mapping[base_col].append(source_col)

    for base_col, source_cols in mapping.items():

        if not source_cols:
            return None

        source_cols.sort()

        expected = list(
            range(
                source_cols[0],
                source_cols[-1] + 1,
            )
        )

        if source_cols != expected:
            return None

        source_positions = [
            source_col
            for base_col in sorted(mapping)
            for source_col in mapping[base_col]
        ]

        if source_positions != sorted(source_positions):
            return None

    return {
        base_col: tuple(source_cols)
        for base_col, source_cols in mapping.items()
    }


def _rows_structure_similar(
        a_row: list[TableCell],
        b_row: list[TableCell],
        *,
        x_tolerance: float = 10.0,
) -> bool:
    if not a_row or not b_row:
        return False

    a_cells = sorted(
        a_row,
        key=lambda cell: int(cell.col),
    )
    b_cells = sorted(
        b_row,
        key=lambda cell: int(cell.col),
    )

    if len(a_cells) != len(b_cells):
        return False

    for a_cell, b_cell in zip(a_cells, b_cells):

        if int(a_cell.col) != int(b_cell.col):
            return False

        if max(int(a_cell.colspan), 1) != max(
                int(b_cell.colspan),
                1,
        ):
            return False

        if a_cell.bbox is None or b_cell.bbox is None:
            continue

        ax1, _, ax2, _ = a_cell.bbox
        bx1, _, bx2, _ = b_cell.bbox

        if abs(ax1 - bx1) > x_tolerance:
            return False

        if abs(ax2 - bx2) > x_tolerance:
            return False

    return True


def _build_logical_rows(rows: list[TableRow]) -> list[dict[int, tuple[TableCell, bool]]]:
    if not rows:
        return []

    max_row = 0
    for row in rows:
        for cell in row.cells:
            start_row = int(cell.row)
            rowspan = max(int(cell.rowspan), 1)

            max_row = max(
                max_row,
                start_row + rowspan - 1,
            )

    if max_row <= 0:
        return []

    # tuple[TableCell, bool] --> table_cell, is_origin
    occupancy: list[dict[int, tuple[TableCell, bool]]] = [
        {}
        for _ in range(max_row + 1)
    ]

    for row in rows:
        for cell in row.cells:
            start_row = int(cell.row)
            start_col = int(cell.col)

            rowspan = max(
                int(cell.rowspan),
                1,
            )

            colspan = max(
                int(cell.colspan),
                1,
            )

            if start_row < 1:
                raise ValueError(
                    f"Invalid cell.row={start_row}: "
                    f"{cell!r}"
                )

            if start_col < 1:
                raise ValueError(
                    f"Invalid cell.col={start_col}: "
                    f"{cell!r}"
                )

            for logical_row in range(
                    start_row,
                    start_row + rowspan,
            ):

                if logical_row >= len(occupancy):
                    raise ValueError(
                        "Cell exceeds table row range: "
                        f"row={logical_row}, "
                        f"cell={cell!r}"
                    )

                for logical_col in range(start_col, start_col + colspan):
                    occupancy[logical_row][logical_col] = (cell, logical_row == cell.row)

    return occupancy[1:]


def _is_rowspan_only_row(row: dict[int, tuple[TableCell, bool]]) -> bool:
    if not row:
        return False

    return all(
        not is_new_cell
        for _, is_new_cell in row.values()
    )


def _merge_single_row(
        logical_row: dict[int, tuple[TableCell, bool]],
        column_mapping: dict[int, tuple[int, ...]],
        base_column_count: int,
) -> list[TableCell]:
    result: list[TableCell] = []

    for base_col in range(
            1,
            base_column_count + 1,
    ):

        source_cols = column_mapping.get(base_col)
        if not source_cols:
            continue

        source_cells = _get_source_cells(
            logical_row=logical_row,
            source_cols=source_cols,
        )

        if not source_cells:
            continue

        merged_cell = _merge_source_cells(
            source_cells=source_cells,
            base_col=base_col,
        )

        result.append(merged_cell)

    return result


def _get_source_cells(
        logical_row: dict[int, tuple[TableCell, bool]], source_cols: tuple[int, ...]
) -> list[tuple[TableCell, bool]]:
    result: list[tuple[TableCell, bool]] = []

    seen: set[int] = set()

    for source_col in source_cols:
        cell, is_original = logical_row.get(source_col, (None, False))
        if cell is None:
            continue

        cell_id = id(cell)

        if cell_id in seen:
            continue

        seen.add(cell_id)
        result.append((cell, is_original))

    return result


def _merge_source_cells(source_cells: list[tuple[TableCell, bool]], base_col: int) -> TableCell:
    if not source_cells:
        raise ValueError(
            "source_cells cannot be empty."
        )

    if len(source_cells) == 1:
        cell, is_original_rowspan = source_cells[0]

        return _copy_cell(
            cell,
            col=base_col,
            colspan=1,
            is_original_rowspan=is_original_rowspan,
        )

    source_cells = sorted(
        source_cells,
        key=lambda cell: int(cell[0].col),
    )

    text = _merge_cell_text(source_cells)
    bbox = _merge_cell_bbox(source_cells)

    rowspan = max(
        max(int(cell.rowspan), 1)
        for cell, _ in source_cells
    )

    first, is_original_rowspan = source_cells[0]

    return _copy_cell(
        first,
        text=text,
        col=base_col,
        colspan=1,
        rowspan=rowspan,
        bbox=bbox,
        is_original_rowspan=is_original_rowspan
    )


def _merge_cell_text(cells: list[tuple[TableCell, bool]]) -> str:
    texts: list[str] = []

    for cell, _ in cells:
        text = cell.text
        text = str(text).strip()

        if not text:
            continue

        texts.append(text)

    return " ".join(texts)


def _merge_cell_bbox(
        cells: list[tuple[TableCell, bool]],
) -> tuple[float, float, float, float] | None:
    bboxes = [
        cell.bbox
        for cell, _ in cells
        if cell.bbox is not None
    ]

    if not bboxes:
        return None

    x1 = min(
        bbox[0]
        for bbox in bboxes
    )

    y1 = min(
        bbox[1]
        for bbox in bboxes
    )

    x2 = max(
        bbox[2]
        for bbox in bboxes
    )

    y2 = max(
        bbox[3]
        for bbox in bboxes
    )

    return x1, y1, x2, y2


def _copy_cell(cell: TableCell, **changes) -> TableCell:
    return cell.model_copy(
        update=changes,
        deep=True,
    )


def merge_rows(
        rows: list[TableRow], column_mapping: dict[int, tuple[int, ...]], same_header: bool = False
) -> list[TableRow]:
    if not rows:
        return []

    # rows = rows[1:] if len(rows) > 1 and same_header else rows
    logical_rows = _build_logical_rows(rows)
    results: list[TableRow] = []
    base_column_count = max(column_mapping)

    current_row: list[TableCell] | None = None
    over = False
    for logical_row in logical_rows:
        if _is_rowspan_only_row(logical_row):
            continue

        single_row: list[TableCell] = [table_cell for table_cell, _ in logical_row.values()]

        if over:
            results.append(TableRow(cells=single_row))
            continue

        if current_row is None:
            current_row = single_row
        else:
            if not _rows_structure_similar(current_row, single_row):
                results.append(TableRow(cells=single_row))
                over = True
                continue

        merged_row = _merge_single_row(
            logical_row=logical_row,
            column_mapping=column_mapping,
            base_column_count=base_column_count,
        )

        if merged_row:
            # table_row: list[TableCell] = []
            # for cell in merged_row:
            #     if cell.is_original_rowspan:
            #         table_row.append(cell)

            results.append(TableRow(cells=merged_row))

    return results[1:] if len(results) > 1 and same_header else results


def merge_table(table_list: list[Table]) -> Table | None:
    if len(table_list) == 0:
        return None

    first_table = table_list[0]
    width = first_table.width
    height = first_table.height

    for table_item in table_list[1:]:
        width = max(width, table_item.width)

        all_bbox = [
            cell.bbox
            for rows in table_item.rows for cell in rows.cells
            if cell.bbox is not None
        ]

        if all_bbox:
            box_array = np.array(all_bbox)
            max_y2 = np.max((box_array[:, 3]))
            height = max(height, height + float(max_y2))

        first_table.rows.extend(table_item.rows)

    first_table.width = width
    first_table.height = height

    occupancy: list[TableRow] = [
        TableRow(cells=[])
        for _ in range(len(first_table.rows))
    ]

    max_cell_count = max(
        (len(row.cells) for row in first_table.rows),
        default=0,
    )

    for cell_index in range(max_cell_count):
        current_cell: TableCell | None = None

        for row_index, row in enumerate(first_table.rows):
            if cell_index >= len(row.cells):
                continue

            cell = row.cells[cell_index]

            if cell.is_original_rowspan:
                current_cell = cell
                occupancy[row_index].cells.append(cell)
            else:
                if current_cell is not None:
                    current_cell.rowspan += 1

    first_table.rows = occupancy

    return first_table
