from __future__ import annotations

from lxml import etree
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal


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


class TableRow(BaseModel):
    cells: list[TableCell] = Field(default_factory=list)


class Table(BaseModel):
    rows: list[TableRow] = Field(
        default_factory=list
    )
    caption: str | None = None
    id: str | None = None

    def add_row(self, *cells: TableCell) -> TableRow:
        row = TableRow(
            cells=list(cells)
        )

        self.rows.append(row)
        return row

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
    def _render_cell(tr: etree._Element, cell: TableCell) -> None:

        td = etree.SubElement(tr, cell.tag)
        # rowspan
        if cell.rowspan > 1:
            td.set("rowspan", str(cell.rowspan))

        # colspan
        if cell.colspan > 1:
            td.set("colspan", str(cell.colspan), )

        td.text = cell.text
