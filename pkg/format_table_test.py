import unittest
from pkg.format_table import Table, TableCell


class FormatTable(unittest.TestCase):
    def test_format_table1(self):
        table = Table(width=600, height=200, caption="学生成绩表")

        table.add_row(
            TableCell(text="姓名", tag="th", bbox=[0,0,200,200]),
            TableCell(text="语文", tag="th", bbox=[200,0,300,100]),
            TableCell(text="数学", tag="th", bbox=[300,0,600,100]),
        )

        table.add_row(
            TableCell(text="90", bbox=[200,100,300,200]),
            TableCell(text="95", bbox=[300,100,600,200]),
        )

        table.calculate_spans(
            tolerance=5
        )

        for cell in table.get_cells():
            print(
                f"text={cell.text!r}, "
                f"row={cell.row}, "
                f"col={cell.col}, "
                f"rowspan={cell.rowspan}, "
                f"colspan={cell.colspan}"
            )


        print(table.to_html())

    def test_format_table2(self):
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
