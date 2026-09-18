import unittest
import os
import json
from easy_to_markdown import pkg
from easy_to_markdown.pkg.format_table import (
    Table, TableCell, compare_header, build_row_mapping, merge_rows)
from easy_to_markdown.generate.to_md import MarkdownFileResult


class FormatTable(unittest.TestCase):
    def test_format_table1(self):
        table = Table(width=600, height=200, caption="学生成绩表")

        table.add_row(
            cells=[TableCell(text="姓名", tag="th", bbox=[0, 0, 200, 200]),
                   TableCell(text="语文", tag="th", bbox=[200, 0, 300, 100]),
                   TableCell(text="数学", tag="th", bbox=[300, 0, 600, 100])]
        )

        table.add_row(
            cells=[TableCell(text="90", bbox=[200, 100, 300, 200]),
                   TableCell(text="95", bbox=[300, 100, 600, 200])]
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

    def test_compare_header(self):
        with open(os.path.join(pkg.MDDir, "md_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        markdown_file_result = MarkdownFileResult.model_validate(data)
        a_children = markdown_file_result.children[1]
        a_table = None
        for item in a_children:
            if item.block_label == "table":
                a_table = item.table_info
                break

        b_children = markdown_file_result.children[2]
        b_table = None
        for item in b_children:
            if item.block_label == "table":
                b_table = item.table_info
                break

        if a_table is None or b_table is None:
            return

        compare_header_result = compare_header(a_table, b_table)
        if compare_header_result is not None:
            merge_rows_result = merge_rows(b_table.rows, compare_header_result, True)
            print(f"{merge_rows_result}")
            b_table.rows = merge_rows_result




    def test_build_row_mapping(self):
        with open(os.path.join(pkg.MDDir, "md_result.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        markdown_file_result = MarkdownFileResult.model_validate(data)
        a_children = markdown_file_result.children[1]
        a_table = None
        for item in a_children:
            if item.block_label == "table":
                a_table = item.table_info
                break

        b_children = markdown_file_result.children[2]
        b_table = None
        for item in b_children:
            if item.block_label == "table":
                b_table = item.table_info
                break

        if a_table is None or b_table is None:
            return

        first_row = a_table.rows[0]
        last_row = a_table.rows[-1]

        row_mapping = build_row_mapping(first_row.cells, last_row.cells)
        print(row_mapping)

        if row_mapping is None:
            return

        merge_rows_result = merge_rows(b_table.rows, row_mapping, False)
        print(merge_rows_result)
