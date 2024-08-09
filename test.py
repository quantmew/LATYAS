from latyas.layout.block import Block, BlockType
from latyas.layout.models.ultralytics.ultralytics_layout_model import UltralyticsLayoutModel

model = UltralyticsLayoutModel.from_pretrained("XiaHan19/360LayoutAnalysis-general6-8n")

from latyas.ocr.models.easyocr.easyocr_ocr_config import EasyOCROCRConfig
from latyas.ocr.models.easyocr.easyocr_ocr_model import EasyOCROCRModel
ocr_model = EasyOCROCRModel(EasyOCROCRConfig())


import pdf2image
import cv2
import numpy as np
import os
file_path = "report2.pdf"
pages = pdf2image.convert_from_path(file_path)
for page_number, page in enumerate(pages):
    page_img = np.asarray(page)
    page_shape = page_img.shape # (h, w, c)
    sf = (page_shape[0] // 256, page_shape[1] // 3)

    page_layout = model.detect(page_img)
    
    vis = page_layout.visualize()
    cv2.imwrite(f'./outputs/output_{page_number}.png', vis)

    # Sort Blocks
    sorted_bbox = []
    for bbox_i in range(len(page_layout)):
        x, y, x2, y2 = page_layout[bbox_i].shape.boundingbox
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
        sorted_bbox.append(((x//sf[1], y//sf[0], x2//sf[1], y2//sf[0]), bbox_i))
    sorted_bbox = sorted(sorted_bbox, key=lambda x: x[0])

    # OCR
    with open(f'./outputs/output_{page_number}.txt', "w", encoding="utf-8") as f:
        for k, bbox_i in sorted_bbox:
            block = page_layout[bbox_i]
            if block.kind in (BlockType.Text, BlockType.Text):
                text = ocr_model.detect(page_layout.crop_image(block))
                # text = text.replace("\n", "")
                f.write(text)
                f.write("\n\n\n\n")