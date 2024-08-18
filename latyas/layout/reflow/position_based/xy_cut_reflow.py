"""
See also: Layout Analysis of Complex Documents.
See also: https://stackoverflow.com/questions/27549574/algorithms-to-extract-text-from-a-pdf-re-flowing-text-layout-from-a-jumble-of-w
"""

import sys
from typing import List, Tuple, Union
from latyas.layout.block import Block
from latyas.layout.layout import Layout

EPSILON = 5


def simple_position_reflow(
    page_layout: Union[Layout, List[Block]], bboxs: List[int]
) -> List[int]:
    # Sort Blocks
    sorted_bbox = []
    min_x, min_y, max_x, max_y = page_layout[0].shape.boundingbox
    for bbox_i in bboxs:
        x, y, x2, y2 = page_layout[bbox_i].shape.boundingbox
        min_x = min(min_x, x)
        max_x = max(max_x, x2)
        min_y = min(min_y, y)
        max_y = max(max_y, y2)

    w = max_x - min_x
    h = max_y - min_y
    sf = (h // 16, w // 3)

    for bbox_i in bboxs:
        x, y, x2, y2 = page_layout[bbox_i].shape.boundingbox
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
        sorted_bbox.append(((x // sf[1], y // sf[0], x2 // sf[1], y2 // sf[0]), bbox_i))
    sorted_bbox = sorted(sorted_bbox, key=lambda x: x[0])

    return [item[1] for item in sorted_bbox]


def horizontal_overlap(
    page_layout: Union[Layout, List[Block]], bboxs: List[int], split: float
) -> Tuple[List[int], List[int], List[int]]:
    left_box = []
    right_box = []
    overlap_box = []

    for bbox_j in bboxs:
        bbox = page_layout[bbox_j].shape.boundingbox
        bbox_start_x = bbox[0]
        bbox_end_x = bbox[2]

        # Check for overlap based on the split value
        if bbox_start_x <= split and bbox_end_x <= split:
            left_box.append(bbox_j)
        elif bbox_start_x >= split and bbox_end_x >= split:
            right_box.append(bbox_j)
        else:
            overlap_box.append(bbox_j)
    return left_box, right_box, overlap_box


def horizontal_region(
    page_layout: Union[Layout, List[Block]],
    bboxs: List[int],
    margin: float = 0.0,
    depth: int = 0,
    max_depth: int = 4,
) -> List[int]:
    """
    水平排布的block
    """
    if len(bboxs) <= 1:
        return bboxs
    if depth > max_depth:
        return simple_position_reflow(page_layout, bboxs)

    possible_x_list = []
    for bbox_i in bboxs:
        bbox = page_layout[bbox_i].shape.boundingbox
        possible_x_list.extend([bbox[0] - margin, bbox[2] + margin])
    possible_x_list = sorted(possible_x_list)

    sorted_bboxs = []
    rest_bboxs = bboxs
    for possible_x in possible_x_list:
        l, r, o = horizontal_overlap(page_layout, rest_bboxs, possible_x)
        if len(o) == 0:
            if len(l) == 0:
                continue

            # print(f"possible x: {possible_x}")
            sorted_bboxs.extend(
                vertical_region(
                    page_layout,
                    l,
                    margin=margin - depth * (margin / max_depth),
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
            rest_bboxs = r
    sorted_bboxs.extend(
        vertical_region(
            page_layout,
            rest_bboxs,
            margin=margin - depth * (margin / max_depth),
            depth=depth + 1,
            max_depth=max_depth,
        )
    )
    return sorted_bboxs


def vertical_overlap(
    page_layout: Union[Layout, List[Block]], bboxs: List[int], split: float
) -> Tuple[List[int], List[int], List[int]]:
    top_box = []
    bottom_box = []
    overlap_box = []

    for bbox_j in bboxs:
        bbox = page_layout[bbox_j].shape.boundingbox
        bbox_start_y = bbox[1]
        bbox_end_y = bbox[3]

        # Check for overlap based on the split value
        if bbox_start_y <= split and bbox_end_y <= split:
            top_box.append(bbox_j)
        elif bbox_start_y >= split and bbox_end_y >= split:
            bottom_box.append(bbox_j)
        else:
            overlap_box.append(bbox_j)
    return top_box, bottom_box, overlap_box


def vertical_region(
    page_layout: Union[Layout, List[Block]],
    bboxs: List[int],
    margin: float = 0.0,
    depth: int = 0,
    max_depth: int = 4,
) -> List[int]:
    """
    垂直排布的block
    """
    if len(bboxs) <= 1:
        return bboxs
    if depth > max_depth:
        return simple_position_reflow(page_layout, bboxs)

    possible_y_list = []
    for bbox_i in bboxs:
        bbox = page_layout[bbox_i].shape.boundingbox
        possible_y_list.extend([bbox[1] - margin, bbox[3] + margin])
    possible_y_list = sorted(possible_y_list)

    sorted_bboxs = []
    rest_bboxs = bboxs
    for possible_y in possible_y_list:
        t, b, o = vertical_overlap(page_layout, rest_bboxs, possible_y)
        if len(o) == 0:
            if len(t) == 0:
                continue

            # print(f"possible y: {possible_y}")
            sorted_bboxs.extend(
                horizontal_region(
                    page_layout,
                    t,
                    margin=margin - depth * (margin / max_depth),
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
            rest_bboxs = b

    sorted_bboxs.extend(
        horizontal_region(
            page_layout,
            rest_bboxs,
            margin=margin - depth * (margin / max_depth),
            depth=depth + 1,
            max_depth=max_depth,
        )
    )
    return sorted_bboxs


def xy_cut_reflow(
    page_layout: Union[Layout, List[Block]], margin: float = 10
) -> List[int]:
    # page_img = page_layout._page
    # page_shape = page_img.shape  # (h, w, c)
    bboxs = []
    for bbox_i in range(len(page_layout)):
        bboxs.append(bbox_i)

    out = horizontal_region(page_layout, bboxs, margin=margin, depth=0, max_depth=8)
    return out
