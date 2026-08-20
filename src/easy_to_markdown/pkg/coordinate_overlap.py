import numpy as np
from pydantic import BaseModel


class OverlapInfo(BaseModel):
    index: int
    overlap: list[int]


def get_overlap_result_np(coordinate: list[list[float]], no_duplicate: bool = False) -> list[OverlapInfo]:
    boxes = np.array(coordinate, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    overlap_matrix = (
            (x2[:, None] > x1[None, :]) &
            (x2[None, :] > x1[:, None]) &
            (y2[:, None] > y1[None, :]) &
            (y2[None, :] > y1[:, None])
    )

    if no_duplicate:
        overlap_matrix = np.triu(overlap_matrix, k=1)
    else:
        np.fill_diagonal(overlap_matrix, False)

    result = [
        OverlapInfo(index=i, overlap=np.where(overlap_matrix[i])[0].tolist())
        for i in range(len(coordinate))
    ]



    return result