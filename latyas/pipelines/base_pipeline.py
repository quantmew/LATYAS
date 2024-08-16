import numpy as np
import pypdfium2
from typing import Dict, Optional
from latyas.layout.block import BlockType
from latyas.layout.layout import Layout
from latyas.layout.models.layout_model import LayoutModel
from latyas.layout.reflow.position_based.xy_cut_reflow import xy_cut_reflow
from latyas.layout.shape import Rectangle
from latyas.ocr.models.ocr_model import OCRModel
from latyas.utils.text_utils import levenshtein_distance


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

    def analyze_pdf(self, page: pypdfium2.PdfPage, threshold:float=0.2) -> Layout:
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

        # Equation OCR
        embedeq_blocks = []
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if block.kind not in (BlockType.Equation, BlockType.EmbedEq):
                continue
            if block.kind == BlockType.EmbedEq:
                if BlockType.EmbedEq in self._ocr_rule:
                    model_name = self._ocr_rule[BlockType.EmbedEq]
                    text = self._ocr_models[model_name].detect(page_layout.crop_image(block))
                    page_layout[bbox_i].set_text(text)
                    embedeq_blocks.append(block)
                else:
                    raise Exception(f"Cannot find the OCR model for {block.kind.name}")
            elif block.kind == BlockType.Equation:
                if BlockType.Equation in self._ocr_rule:
                    model_name = self._ocr_rule[BlockType.Equation]
                    text = self._ocr_models[model_name].detect(page_layout.crop_image(block))
                    page_layout[bbox_i].set_text(text)
                else:
                    raise Exception(f"Cannot find the OCR model for {block.kind.name}")

        # Text OCR
        textpage = page.get_textpage()
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if block.kind not in (BlockType.Text, BlockType.Title, BlockType.Caption):
                continue
            x1, y1, x2, y2 = block.shape.boundingbox
            x_1, y_1, x_2, y_2 = x1 / rs, height - y2 / rs, x2 / rs, height - y1 / rs
            if block.kind in self._ocr_rule:
                model_name = self._ocr_rule[block.kind]
                ocr_text = self._ocr_models[model_name].detect(page_layout.crop_image(block))
            else:
                raise Exception(f"Cannot find the OCR model for {block.kind.name}")
            pdf_text = get_text_by_bbox(textpage, x_1, y_1, x_2, y_2)

            dis = levenshtein_distance(ocr_text, pdf_text)
            dis_percent = dis / max(len(ocr_text), len(pdf_text))

            if dis_percent < threshold:
                text = pdf_text
            else:
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

        # Equation OCR
        embedeq_blocks = []
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if block.kind not in (BlockType.Equation, BlockType.EmbedEq):
                continue
            if block.kind == BlockType.EmbedEq:
                if BlockType.EmbedEq in self._ocr_rule:
                    model_name = self._ocr_rule[BlockType.EmbedEq]
                    text = self._ocr_models[model_name].detect(page_layout.crop_image(block))
                    page_layout[bbox_i].set_text(text)
                    embedeq_blocks.append(block)
                else:
                    raise Exception(f"Cannot find the OCR model for {block.kind.name}")
            elif block.kind == BlockType.Equation:
                if BlockType.Equation in self._ocr_rule:
                    model_name = self._ocr_rule[BlockType.Equation]
                    text = self._ocr_models[model_name].detect(page_layout.crop_image(block))
                    page_layout[bbox_i].set_text(text)
                else:
                    raise Exception(f"Cannot find the OCR model for {block.kind.name}")

        # Text OCR
        for bbox_i in range(len(page_layout)):
            block = page_layout[bbox_i]
            if block.kind not in (BlockType.Text, BlockType.Title, BlockType.Caption):
                continue
            if block.kind in self._ocr_rule:
                model_name = self._ocr_rule[block.kind]
                text = self._ocr_models[model_name].detect(page_layout.crop_image(block))
            else:
                raise Exception(f"Cannot find the OCR model for {block.kind.name}")
            block.set_text(text)

        # Reflow
        sorted_block_indices = xy_cut_reflow(page_layout)
        page_layout._blocks = [page_layout._blocks[i] for i in sorted_block_indices]

        return page_layout