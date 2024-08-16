from typing import List
from latyas.layout.block import Block, BlockType
from latyas.layout.models.ultralytics.ultralytics_layout_model import (
    UltralyticsLayoutModel,
)
from latyas.layout.sorting.position_based.position_sorting import position_sorting
from latyas.layout.sorting.semantic_based.bert_sorting import bert_sorting
from latyas.utils.text_utils import levenshtein_distance

model = UltralyticsLayoutModel.from_pretrained("XiaHan19/360LayoutAnalysis-general6-8n")

# from latyas.ocr.models.easyocr.easyocr_ocr_config import EasyOCROCRConfig
# from latyas.ocr.models.easyocr.easyocr_ocr_model import EasyOCROCRModel
# ocr_model = EasyOCROCRModel(EasyOCROCRConfig())

from latyas.ocr.models.paddleocr.paddleocr_ocr_config import PaddleOCRConfig
from latyas.ocr.models.paddleocr.paddleocr_ocr_model import PaddleOCRModel
from latyas.layout.shape import Rectangle

ocr_model = PaddleOCRModel(PaddleOCRConfig())

import pypdfium2
import cv2
import numpy as np
import os
import tqdm


def get_text_by_bbox(textpage, x_1, y_1, x_2, y_2) -> str:
    
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
                left=rect_cord[0], bottom=rect_cord[1], right=rect_cord[2], top=rect_cord[3]
            ).replace("\n", "")
            pdf_text += pdf_text_line
    return pdf_text


def get_page_text(page_number: int, page: pypdfium2.PdfPage) -> List[str]:
    width, height = page.get_size()
    render_scale = rs = 2
    bitmap = page.render(
        scale=render_scale,  # 72dpi resolution
        rotation=0,  # no additional rotation
    )
    pil_image = bitmap.to_pil()

    page_img = np.asarray(pil_image)
    page_layout = model.detect(page_img)

    # OCR
    text_list = []
    textpage = page.get_textpage()
    for bbox_i in range(len(page_layout)):
        block = page_layout[bbox_i]
        if block.kind not in (BlockType.Text, BlockType.Title, BlockType.Caption):
            continue
        x1, y1, x2, y2 = block.shape.boundingbox
        x_1, y_1, x_2, y_2 = x1 / rs, height - y2 / rs, x2 / rs, height - y1 / rs
        ocr_text = ocr_model.detect(page_layout.crop_image(block))
        pdf_text = get_text_by_bbox(textpage, x_1, y_1, x_2, y_2)
        print("=========== OCR Text =============")
        print(ocr_text)
        print("=========== PDF Text =============")
        print(pdf_text)
        dis = levenshtein_distance(ocr_text, pdf_text)
        dis_percent = dis / max(len(ocr_text), len(pdf_text))
        print(f"编辑距离：{dis_percent}")

        if dis_percent < 0.2:
            text = pdf_text
        else:
            text = ocr_text
        if text.startswith("图") or text.startswith("表"):
            continue
        if len(text) < 256 and ("见表" in text or "见图" in text):
            continue
        if block.kind in (BlockType.Title, BlockType.Caption):
            text_list.append("\n==========\n")
        else:
            text_list.append("\n")
        block.set_text(text)
        text_list.append(text)
    textpage.close()
    
    sorted_block_indices = position_sorting(page_layout)
    page_layout._blocks = [page_layout._blocks[i] for i in sorted_block_indices]

    # write images
    vis = page_layout.visualize()
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"./outputs/output_{page_number}.jpg", vis)
    return text_list

def main():
    file_path = "report3.pdf"
    pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
    try:
        texts = []
        for page_number, page in enumerate(pdf_reader):
            text = get_page_text(page_number, page)
            texts.append(text)
            page.close()
        with open(f"./outputs/text_output.txt", "w", encoding="utf-8") as f:
            for text in texts:
                f.write("".join(text))
    finally:
        pdf_reader.close()


if __name__ == "__main__":
    main()
