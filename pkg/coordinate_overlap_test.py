import unittest
from pkg.coordinate_overlap import get_overlap_result_np


class TestOverlap(unittest.IsolatedAsyncioTestCase):
    def test_coordinate_overlap(self):
        coordinate = [
            [
                1.542023777961731,
                0.03684183955192566,
                934,
                37
            ],
            [
                1.842023777961731,
                0.04684183955192566,
                935,
                38
            ],
            [
                1.542023777961731,
                0.03684183955192566,
                934,
                37
            ],
            [
                1.542023777961731,
                0.03684183955192566,
                934,
                37
            ],
            [
                15.309467315673828,
                4.3382134437561035,
                133.2974853515625,
                36.20841598510742
            ],
            [
                14.329008102416992,
                4.173314571380615,
                133.60044860839844,
                36.77362823486328
            ]
        ]
        results = get_overlap_result_np(coordinate=coordinate, no_duplicate=True)
        print(results)

