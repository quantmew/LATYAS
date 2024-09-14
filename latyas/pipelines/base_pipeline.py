import cv2
import numpy as np
import pypdfium2
from typing import Dict, List, Optional, Tuple, Union
from latyas.layout.block import Block, BlockType, is_text_block
from latyas.layout.layout import Layout
from latyas.layout.models.layout_model import LayoutModel
from latyas.layout.reflow.position_based.xy_cut_reflow import xy_cut_reflow
from latyas.layout.shape import Rectangle
from latyas.ocr.models.ocr_model import OCRModel
from latyas.ocr.text_bbox import TextBoundingBox
from latyas.utils.text_utils import levenshtein_distance

def coord_latyas_to_pdf(x, y, widht, height):
    return x, height - y

def get_text_by_bbox(
    textpage: pypdfium2.PdfPage, x_1: float, y_1: float, x_2: float, y_2: float
) -> str:

    # The page coordinate system's starting point (0,0) is the left-bottom corner of a page.
    # The Y axis is directed from the bottom of the page to the top.
    pdf_text = ""
    rects_n = textpage.count_rects()
    for rect_i in range(rects_n):
        rect_cord = textpage.get_rect(rect_i)
        lhs_rect = Rectangle(rect_cord[0], rect_cord[1], rect_cord[2], rect_cord[3])
        rhs_rect = Rectangle(x_1, y_1, x_2, y_2)

        overlap_area = lhs_rect.intersect(rhs_rect).area / min(
            lhs_rect.area, rhs_rect.area
        )

        if overlap_area > 0.5:
            pdf_text_line = textpage.get_text_bounded(
                left=rect_cord[0],
                bottom=rect_cord[1],
                right=rect_cord[2],
                top=rect_cord[3],
            ).replace("\n", "")
            pdf_text += pdf_text_line
    return pdf_text


class BlockRuleKey(object):
    def __init__(self, kind: BlockType, has_equation: bool = False):
        self._kind = kind
        self._has_equation = has_equation

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        if isinstance(value, BlockType):
            self._kind = value
        else:
            raise TypeError("kind must be a BlockType")

    @property
    def has_equation(self):
        return self._has_equation

    @has_equation.setter
    def has_equation(self, value):
        if isinstance(value, bool):
            self._has_equation = value
        else:
            raise TypeError("has_equation must be a boolean")

    def __str__(self):
        return f"BlockRuleKey(kind={self.kind}, has_equation={self.has_equation})"

    def __repr__(self):
        return f"BlockRuleKey(kind={self.kind}, has_equation={self.has_equation})"


class BasePipeline(object):
    def __init__(self) -> None:
        self._layout_models: Dict[str, LayoutModel] = {}
        self._ocr_models: Dict[str, OCRModel] = {}
        self._ocr_rule: Dict[BlockType, str] = {}

    def add_layout_model(self, name: str, layout_model: LayoutModel) -> None:
        self._layout_models[name] = layout_model

    def add_ocr_model(self, name: str, ocr_model: OCRModel) -> None:
        self._ocr_models[name] = ocr_model

    def add_ocr_rule(self, block_type: BlockType, rule: str) -> None:
        self._ocr_rule[block_type] = rule

    def analyze_pdf(self, page: pypdfium2.PdfPage) -> Layout:
        width, height = page.get_size()
        render_scale = rs = 2
        bitmap = page.render(
            scale=render_scale,  # 72dpi resolution
            rotation=0,  # no additional rotation
        )
        pil_image = bitmap.to_pil()
        page_img = np.asarray(pil_image)

        # Layout Analysis
        page_layout: Optional[Layout] = None
        for layout_model_name, layout_model in self._layout_models.items():
            each_page_layout = layout_model.detect(page_img)
            if page_layout is None:
                page_layout = each_page_layout
            else:
                page_layout.merge(each_page_layout)
        page_layout.remove_overlapping(strategy="merge")

        # Equation OCR
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if block.kind not in (BlockType.Equation,):
                continue
            if block.kind == BlockType.Equation:
                if BlockType.Equation in self._ocr_rule:
                    model_name = self._ocr_rule[BlockType.Equation]
                    text = self._ocr_models[model_name].recognize(page_layout.crop_image(block))
                    page_layout[bbox_i].set_text(text)
                else:
                    raise Exception(f"Cannot find the OCR model for {block.kind.name}")

        # Text with Embed Equation
        for bbox_text_index in range(len(page_layout)):
            text_block = page_layout[bbox_text_index]
            if not is_text_block(text_block.kind):
                continue
            # Check EmbedEq
            has_embed_eq = False
            equations = []
            for bbox_j in range(len(page_layout)):
                if bbox_text_index == bbox_j:
                    continue
                if page_layout[bbox_j].kind == BlockType.EmbedEq and page_layout[
                    bbox_j
                ].shape.is_inside(page_layout[bbox_text_index].shape):
                    has_embed_eq = True
                    equations.append(bbox_j)
            
            if not has_embed_eq:
                continue
            text_block._has_equation = True
            if BlockType.TextWithEquation in self._ocr_rule:
                model_name = self._ocr_rule[BlockType.TextWithEquation]
                text = self._ocr_models[model_name].recognize(page_layout.crop_image(text_block))
                page_layout[bbox_text_index].set_text(text)
            else:
                raise Exception(f"Cannot find the OCR model for {block.kind.name}")
        
        # Table OCR
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if block.kind != BlockType.Table:
                continue

            x1, y1, x2, y2 = block.shape.boundingbox
            if block.kind in self._ocr_rule:
                model_name = self._ocr_rule[block.kind]
                ocr_text = self._ocr_models[model_name].recognize(page_layout.crop_image(block))
            else:
                raise Exception(f"Cannot find the Table OCR model for {block.kind.name}")
            text = ocr_text
            block.set_text(text)
        
        # Text OCR
        textpage = page.get_textpage()
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if not is_text_block(block.kind):
                continue

            if block._has_equation:
                continue
            x1, y1, x2, y2 = block.shape.boundingbox
            if block.kind in self._ocr_rule:
                model_name = self._ocr_rule[block.kind]
                ocr_text = self._ocr_models[model_name].recognize(page_layout.crop_image(block))
            else:
                raise Exception(f"Cannot find the OCR model for {block.kind.name}")
            text = ocr_text
            block.set_text(text)
        textpage.close()

        # Reflow
        sorted_block_indices = xy_cut_reflow(page_layout)
        page_layout._blocks = [page_layout._blocks[i] for i in sorted_block_indices]

        return page_layout

    def analyze_image(self, page_img: np.ndarray) -> Layout:
        height, width =page_img.shape[0], page_img.shape[1]
        
        # Layout Analysis
        page_layout: Optional[Layout] = None
        for layout_model_name, layout_model in self._layout_models.items():
            each_page_layout = layout_model.detect(page_img)
            if page_layout is None:
                page_layout = each_page_layout
            else:
                page_layout.merge(each_page_layout)
        page_layout.remove_overlapping(strategy="merge")

        # Equation OCR
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if block.kind not in (BlockType.Equation,):
                continue
            if block.kind == BlockType.Equation:
                if BlockType.Equation in self._ocr_rule:
                    model_name = self._ocr_rule[BlockType.Equation]
                    text = self._ocr_models[model_name].recognize(page_layout.crop_image(block))
                    page_layout[bbox_i].set_text(text)
                else:
                    raise Exception(f"Cannot find the OCR model for {block.kind.name}")

        # Text with Embed Equation
        for bbox_text_index in range(len(page_layout)):
            text_block = page_layout[bbox_text_index]
            if not is_text_block(text_block.kind):
                continue
            # Check EmbedEq
            has_embed_eq = False
            equations = []
            for bbox_j in range(len(page_layout)):
                if bbox_text_index == bbox_j:
                    continue
                if page_layout[bbox_j].kind == BlockType.EmbedEq and page_layout[
                    bbox_j
                ].shape.is_inside(page_layout[bbox_text_index].shape):
                    has_embed_eq = True
                    equations.append(bbox_j)
            
            if not has_embed_eq:
                continue
            text_block._has_equation = True
            if BlockType.TextWithEquation in self._ocr_rule:
                model_name = self._ocr_rule[BlockType.TextWithEquation]
                text = self._ocr_models[model_name].recognize(page_layout.crop_image(text_block))
                page_layout[bbox_text_index].set_text(text)
            else:
                raise Exception(f"Cannot find the OCR model for {block.kind.name}")

        # Text OCR
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if not is_text_block(block.kind):
                continue

            if block._has_equation:
                continue
            x1, y1, x2, y2 = block.shape.boundingbox
            if block.kind in self._ocr_rule:
                model_name = self._ocr_rule[block.kind]
                ocr_text = self._ocr_models[model_name].recognize(page_layout.crop_image(block))
            else:
                raise Exception(f"Cannot find the OCR model for {block.kind.name}")
            text = ocr_text
            block.set_text(text)

        # Reflow
        sorted_block_indices = xy_cut_reflow(page_layout)
        page_layout._blocks = [page_layout._blocks[i] for i in sorted_block_indices]

        return page_layout
