


from typing import List
from latyas.layout.layout import Layout


def position_reflow(page_layout: Layout, h_chunk: int = 256, w_chunk: int = 3) -> List[int]:
    page_img = page_layout._page
    page_shape = page_img.shape  # (h, w, c)
    sf = (page_shape[0] // h_chunk, page_shape[1] // w_chunk)

    # Sort Blocks
    sorted_bbox = []
    for bbox_i in range(len(page_layout)):
        x, y, x2, y2 = page_layout[bbox_i].shape.boundingbox
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
        sorted_bbox.append(((x // sf[1], y // sf[0], x2 // sf[1], y2 // sf[0]), bbox_i))
    sorted_bbox = sorted(sorted_bbox, key=lambda x: x[0])

    return [item[1] for item in sorted_bbox]
