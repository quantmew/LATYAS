import copy
import warnings
import cv2
import numpy as np
from typing import Generator, List, Optional, Tuple, Union

from latyas.layout.block import BLOCK_TYPE_COLOR_MAP, Block, BlockType
from latyas.layout.shape import Rectangle


class Layout(object):
    def __init__(
        self, blocks: Optional[List] = None, page: Optional[np.ndarray] = None
    ) -> None:
        if blocks is None:
            blocks = []
        self._blocks: List[Block] = blocks
        # TODO: if page is None
        self._page: np.ndarray = page
    
    @property
    def height(self) -> int:
        return self._page.shape[0]

    @property
    def width(self) -> int:
        return self._page.shape[1]

    def __getitem__(self, key: Union[int, slice]) -> Block:
        blocks = self._blocks[key]
        if isinstance(key, slice):
            return self.__class__(self._blocks[key], self._page)
        else:
            return blocks

    def __setitem__(self, key: int, newvalue: Block):
        self._blocks[key] = newvalue

    def __delitem__(self, key):
        del self._blocks[key]

    def __len__(self) -> int:
        return len(self._blocks)

    def __iter__(self) -> Generator[Block, None, None]:
        for ele in self._blocks:
            yield ele

    def __repr__(self) -> str:
        info_str = ", ".join([f"{key}={val}" for key, val in vars(self).items()])
        return f"{self.__class__.__name__}({info_str})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Layout):
            return self._blocks == other._blocks
        else:
            return False
    
    def copy(self) -> "Layout":
        return Layout(
            blocks=copy.deepcopy(self._blocks),
            page=self._page.copy()
        )
    
    def insert(self, key: int, value: Block):
        self._blocks.insert(key, value)

    def merge(self, other: "Layout"):
        self._blocks.extend(other._blocks)
    
    def page_sort(self, reverse=False):
        """
        Deprecated
        """
        warnings.warn("deprecated", DeprecationWarning)
        sf = (self._page.shape[0] // 256, self._page.shape[1] // 3)
        sorted_bbox = []
        for bbox_i in range(len(self._blocks)):
            x, y, x2, y2 = self._blocks[bbox_i].shape.boundingbox
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
            sorted_bbox.append(
                ((x // sf[1], y // sf[0], x2 // sf[1], y2 // sf[0]), bbox_i)
            )
        sorted_bbox = sorted(sorted_bbox, key=lambda x: x[0], reverse=reverse)
        self._blocks = [self._blocks[bbox_i] for k, bbox_i in sorted_bbox]

    def remove_overlapping(self, area_threshold=0.5, keep_larger=True):
        to_remove = []
        for block_i in range(len(self._blocks)):
            for block_j in range(block_i + 1, len(self._blocks)):
                block_lhs = self._blocks[block_i]
                block_rhs = self._blocks[block_j]
                if not isinstance(block_lhs.shape, Rectangle):
                    continue
                if not isinstance(block_rhs.shape, Rectangle):
                    continue

                intersect_rect = block_lhs.shape.intersect(block_rhs.shape)
                if (
                    intersect_rect.area > area_threshold * block_lhs.shape.area
                    or intersect_rect.area > area_threshold * block_rhs.shape.area
                ):
                    if keep_larger:
                        if block_lhs.shape.area > block_rhs.shape.area:
                            to_remove.append(block_j)
                        else:
                            to_remove.append(block_i)
                    else:
                        if block_lhs.shape.area < block_rhs.shape.area:
                            to_remove.append(block_j)
                        else:
                            to_remove.append(block_i)
        to_remove = sorted(list(set(to_remove)))
        for block_i in reversed(to_remove):
            self._blocks.pop(block_i)

    def crop_image(self, block: Block) -> Optional[np.ndarray]:
        if self._page is None:
            return None
        x1, y1, x2, y2 = block.shape.boundingbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox_image = self._page[y1:y2, x1:x2]
        return bbox_image
    
    def mask_image(self, block: Block, color: Union[str, Tuple[int, int, int]] = (255, 255, 255)):
        if self._page is None:
            return None
        x1, y1, x2, y2 = block.shape.boundingbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        self._page[y1:y2, x1:x2, :] = color # TODO: support hex color

    def visualize(self, thickness=2) -> np.ndarray:
        vis = self._page.copy()
        for block_i, block in enumerate(self._blocks):
            if not isinstance(block.shape, Rectangle):
                continue
            x_1 = int(block.shape.x_1)
            x_2 = int(block.shape.x_2)
            y_1 = int(block.shape.y_1)
            y_2 = int(block.shape.y_2)
            cv2.rectangle(
                vis,
                (x_1, y_1),
                (x_2, y_2),
                BLOCK_TYPE_COLOR_MAP[block.kind],
                thickness,
            )
            # Add text label for element type in the top-right corner of the bounding box
            cv2.putText(
                vis,
                f"{block.kind.name}-{block_i}",
                (x_2, y_1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BLOCK_TYPE_COLOR_MAP[block.kind],
                1,
            )
        return vis
