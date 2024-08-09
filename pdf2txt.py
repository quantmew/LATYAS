from typing import List
from latyas.layout.block import Block, BlockType
from latyas.layout.models.ultralytics.ultralytics_layout_model import (
    UltralyticsLayoutModel,
)
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


def get_page_text(page_number: int, page: pypdfium2.PdfPage) -> List[str]:
    width, height = page.get_size()
    render_scale = rs = 2
    bitmap = page.render(
        scale=render_scale,  # 72dpi resolution
        rotation=0,  # no additional rotation
    )
    pil_image = bitmap.to_pil()

    page_img = np.asarray(pil_image)
    page_shape = page_img.shape  # (h, w, c)
    sf = (page_shape[0] // 256, page_shape[1] // 3)

    page_layout = model.detect(page_img)

    vis = page_layout.visualize()
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"./outputs/output_{page_number}.jpg", vis)

    # Sort Blocks
    sorted_bbox = []
    for bbox_i in range(len(page_layout)):
        x, y, x2, y2 = page_layout[bbox_i].shape.boundingbox
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
        sorted_bbox.append(((x // sf[1], y // sf[0], x2 // sf[1], y2 // sf[0]), bbox_i))
    sorted_bbox = sorted(sorted_bbox, key=lambda x: x[0])

    # OCR
    text_list = []
    textpage = page.get_textpage()
    for k, bbox_i in sorted_bbox:
        block = page_layout[bbox_i]
        if block.kind not in (BlockType.Text, BlockType.Title):
            continue
        x1, y1, x2, y2 = block.shape.boundingbox
        x_1, y_1, x_2, y_2 = x1 / rs, height - y2 / rs, x2 / rs, height - y1 / rs

        ocr_text = ocr_model.detect(page_layout.crop_image(block))
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
        text_list.append(text)
    textpage.close()
    return text_list


def main():
    file_path = "report2.pdf"
    pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
    try:
        for page_number, page in enumerate(pdf_reader):
            text = get_page_text(page_number, page)
            with open(f"./outputs/text_{page_number}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(text))
            page.close()
    finally:
        pdf_reader.close()


if __name__ == "__main__":
    main()
