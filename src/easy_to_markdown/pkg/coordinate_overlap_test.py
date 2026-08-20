import unittest
from easy_to_markdown.pkg.coordinate_overlap import get_overlap_result_np


class TestOverlap(unittest.IsolatedAsyncioTestCase):
    def test_coordinate_overlap(self):
        coordinate = [
            [
                9.2938595, 13.232447, 659.21344, 920.61584
            ],
            [
                658.4059, 5.629543, 1307.7343, 933
            ],
            [
                11.747079, 16.504196, 660.16864, 243.23915
            ],
            [
                660.26984, 360.3405, 1306.657, 916.4146
            ],
            [
                661.0359, 15.553273, 1291.4994, 241.76112
            ],
        ]
        results = get_overlap_result_np(coordinate=coordinate, no_duplicate=True)
        print(results)

