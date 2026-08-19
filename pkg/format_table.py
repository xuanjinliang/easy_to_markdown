from __future__ import annotations
from lxml import etree, html
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal
import numpy as np


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
    def render(self, table: Table) -> str:
        table_element = etree.Element("table")

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

        td = etree.SubElement(tr, cell.tag)
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
