from __future__ import annotations
from lxml import etree, html
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal
import numpy as np
import re


class TableCell(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
    )
    text: str = ""
    html: str | None = None
    row: int = Field(default=0, ge=0)
    col: int = Field(default=0, ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    tag: Literal["td", "th"] = "td"
    bbox: Optional[list[float]] = Field(default=None, min_length=4, max_length=4)


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
            )

            cell.row = int(
                row_start[i]
            )

            cell.colspan = int(
                cols[i]
            )

            cell.rowspan = int(
                rows[i]
            )

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

        if cell.html is not None:
            fragments = html.fragments_fromstring(
                cell.html
            )

            for fragment in fragments:
                if isinstance(fragment, str):
                    td.text = fragment
                else:
                    td.append(fragment)
        else:
            # td.text = cell.text
            self.set_text_with_br(td, cell.text)


def table_html_to_cell(table_html: str) -> list[list[TableCell]]:
    root = html.fromstring(table_html)

    table_list: list[list[TableCell]] = []
    for tr in root.xpath(".//tr"):
        row_list: list[TableCell] = []
        for td in tr.xpath("./td | ./th"):
            text = td.text_content().strip()
            rowspan = int(td.get("rowspan", "1"))
            colspan = int(td.get("colspan", "1"))

            row_list.append(TableCell(text=text, tag=td.tag, rowspan=rowspan, colspan=colspan))

        table_list.append(row_list)

    return table_list


def build_table_grid(table: list[list[TableCell]]) -> list[list[TableCell | None]]:
    grid: list[list[TableCell | None]] = []

    for row_index, row in enumerate(table):
        while len(grid) <= row_index:
            grid.append([])

        col_index = 0

        for cell in row:
            while (
                    col_index < len(grid[row_index])
                    and grid[row_index][col_index] is not None
            ):
                col_index += 1

            for r in range(
                    row_index,
                    row_index + cell.rowspan,
            ):
                while len(grid) <= r:
                    grid.append([])

                for c in range(
                        col_index,
                        col_index + cell.colspan,
                ):
                    while len(grid[r]) <= c:
                        grid[r].append(None)

                    grid[r][c] = cell

            col_index += cell.colspan

    return grid


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
