import unittest
from pkg.format_table import Table, TableCell, HtmlTableRenderer


class FormatTable(unittest.TestCase):
    def test_format_table1(self):
        table = Table(caption="学生成绩表", )

        table.add_row(
            TableCell(text="姓名", rowspan=2, tag="th"),
            TableCell(text="成绩", colspan=2, tag="th"),
        )

        table.add_row(
            TableCell(text="语文", tag="th"),
            TableCell(text="数学", tag="th"),
        )

        table.add_row(
            TableCell(text="张三", bbox=[100, 200, 300, 250]),
            TableCell(text="90", bbox=[300, 200, 400, 250]),
            TableCell(text="95", bbox=[400, 200, 500, 250]),
        )

        table.add_row(
            TableCell(text="李四"),
            TableCell(text="88"),
            TableCell(text="92"),
        )
        print(table.to_html())

    def test_format_table2(self):
        table = Table(caption="销售统计", )

        table.add_row(
            TableCell(text="产品", rowspan=2, tag="th"),
            TableCell(text="2026", colspan=2, tag="th"),
            TableCell(text="总计", rowspan=2, tag="th"),
        )

        table.add_row(
            TableCell(text="Q1", tag="th"),
            TableCell(text="Q2", tag="th"),
        )

        table.add_row(
            TableCell(text="A"),
            TableCell(text="100"),
            TableCell(text="120"),
            TableCell(text="220"),
        )

        table.add_row(
            TableCell(text="B"),
            TableCell(text="200"),
            TableCell(text="180"),
            TableCell(text="380"),
        )

        print(table.to_html())

    def test_format_table3(self):
        cell = TableCell(
            text="张三",
            row=2,
            col=0,
            rowspan=1,
            colspan=1,
            bbox=[100, 200, 300, 250]
        )

        print("dict:")
        print(cell.model_dump())
        print("\n")
        # Pydantic -> JSON
        print("json:")
        print(cell.model_dump_json())
