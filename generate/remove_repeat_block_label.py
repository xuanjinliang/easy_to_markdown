import math
from generate import ParsingResult


class RemoveRepeatBlockLabel:
    DEFAULT_LABEL_PRIORITY = {
        "title": 100,
        "isolate_formula": 95,
        "table": 95,
        "figure": 95,
        "plain text": 80,
        "table_caption": 70,
        "figure_caption": 70,
        "formula_caption": 70,
        "table_footnote": 70,
        "abandon": 10,
    }

    SUPPRESS_INNER_LABELS = {
        "title",
        "plain text",
        "abandon"
    }

    SAME_LABEL_IOU_THRESH_BY_LABEL = {
        "figure": 0.90,
        "table": 0.90,
        "isolate_formula": 0.80,
        "table_caption": 0.70,
        "figure_caption": 0.70,
        "formula_caption": 0.70,
        "table_footnote": 0.70,
    }

    def __init__(self,
                 blocks: list[ParsingResult],
                 close_tol: float = 2.0,
                 iou_thresh: float = 0.45,
                 contain_thresh: float = 0.85,
                 same_label_iou_thresh: float = 0.35):
        self.blocks = blocks
        self.close_tol = close_tol
        self.iou_thresh = iou_thresh
        self.contain_thresh = contain_thresh
        self.same_label_iou_thresh = same_label_iou_thresh

    @staticmethod
    def bbox_key(box) -> tuple[int, ...]:
        return tuple(math.ceil(float(x)) for x in box)

    @staticmethod
    def bbox_area(box: list[float]) -> float:
        if box is None or len(box) != 4:
            return 0.0

        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def bbox_close(self, box1: list[float], box2: list[float]) -> bool:
        tol = self.close_tol

        if box1 is None or box2 is None:
            return False

        if len(box1) != 4 or len(box2) != 4:
            return False

        return all(abs(float(a) - float(b)) <= tol for a, b in zip(box1, box2))

    def bbox_intersection(self, box1: list[float], box2: list[float]):
        """
        计算两个 bbox 的交集面积。
        """
        if box1 is None or box2 is None:
            return 0.0

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        return self.bbox_area([x1, y1, x2, y2])

    def bbox_iou(self, box1: list[float], box2: list[float]):
        inter = self.bbox_intersection(box1, box2)
        area1 = self.bbox_area(box1)
        area2 = self.bbox_area(box2)

        union = area1 + area2 - inter
        if union <= 0:
            return 0.0

        return inter / union

    def bbox_overlap_min(self, box1: list[float], box2: list[float]):
        inter = self.bbox_intersection(box1, box2)
        area1 = self.bbox_area(box1)
        area2 = self.bbox_area(box2)

        min_area = min(area1, area2)
        if min_area <= 0:
            return 0.0

        return inter / min_area

    def should_treat_as_duplicate(self, block1: ParsingResult, block2: ParsingResult):
        box1 = block1.block_bbox
        box2 = block2.block_bbox

        if box1 is None or box2 is None:
            return False

        label1 = block1.block_label
        label2 = block2.block_label

        iou = self.bbox_iou(box1, box2)
        overlap_min = self.bbox_overlap_min(box1, box2)

        if (
                label1 == label2 and
                iou > self.SAME_LABEL_IOU_THRESH_BY_LABEL.get(label1, self.same_label_iou_thresh)
        ):
            return True

        elif iou >= self.iou_thresh:
            return True

        if overlap_min >= self.contain_thresh:
            return True

        return False

    def is_better_block(self, new_block: ParsingResult, old_block: ParsingResult):
        label_priority = self.DEFAULT_LABEL_PRIORITY

        new_p = label_priority.get(new_block.block_label, 50)
        old_p = label_priority.get(old_block.block_label, 50)

        new_score = new_block.score
        old_score = old_block.score

        new_area = self.bbox_area(new_block.block_bbox)
        old_area = self.bbox_area(old_block.block_bbox)

        return (new_p, new_score, new_area) > (old_p, old_score, old_area)

    def dedup_exact_same_bbox(self):
        best_by_key: dict[tuple[int, ...], int] = {}

        blocks = self.blocks

        for i, block in enumerate(blocks):
            box = block.block_bbox
            if box is None or block.remove:
                continue

            key = self.bbox_key(box)

            if key not in best_by_key:
                best_by_key[key] = i
            else:
                old = best_by_key[key]
                same_index = i
                if self.is_better_block(block, blocks[old]):
                    same_index = old
                    best_by_key[key] = i

                blocks[same_index].remove = True
                blocks[same_index].remove_reason = f"same_bbox:{blocks[same_index].block_id}"

    def dedup_close_bbox(self):
        kept: list[int] = []
        blocks = self.blocks

        for i, block in enumerate(blocks):
            box = block.block_bbox
            if box is None or block.remove:
                continue

            matched_idx = None

            for j in kept:
                kept_block = blocks[j]
                if self.bbox_close(box, kept_block.block_bbox):
                    matched_idx = j
                    break

            if matched_idx is None:
                kept.append(i)
            else:
                close_index: int = i
                if self.is_better_block(block, blocks[matched_idx]):
                    close_index = matched_idx
                    kept[matched_idx] = i

                blocks[close_index].remove = True
                blocks[close_index].remove_reason = f"close_bbox:{blocks[close_index].block_id}"

    def can_compare_for_duplicate(self, block1: ParsingResult, block2: ParsingResult) -> bool:
        same_label = block1.block_label == block2.block_label

        if same_label:
            return True

        return (
                block1.block_label in self.SUPPRESS_INNER_LABELS
                or block2.block_label in self.SUPPRESS_INNER_LABELS
        )

    def dedup_overlap_blocks(self):
        blocks = self.blocks
        kept: list[int] = []

        for i, block in enumerate(blocks):
            box = block.block_bbox
            if box is None or block.remove:
                continue

            duplicate_idx = None

            for j in kept:
                kept_block = blocks[j]

                if not self.can_compare_for_duplicate(block, kept_block):
                    continue

                if self.should_treat_as_duplicate(
                        block,
                        kept_block
                ):
                    duplicate_idx = j
                    break

            if duplicate_idx is None:
                kept.append(i)
            else:
                overlap_index = i
                if self.is_better_block(block, blocks[duplicate_idx]):
                    overlap_index = duplicate_idx
                    kept[duplicate_idx] = i

                blocks[overlap_index].remove = True
                blocks[overlap_index].remove_reason = f"overlap_bbox:{blocks[overlap_index].block_id}"

    def run(self) -> list[ParsingResult]:
        self.dedup_exact_same_bbox()
        self.dedup_close_bbox()
        self.dedup_overlap_blocks()
        return self.blocks
