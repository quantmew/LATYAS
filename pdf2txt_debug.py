# from latyas.layout.models.ultralytics.ultralytics_layout_model import UltralyticsLayoutModel
# model = UltralyticsLayoutModel.from_pretrained("XiaHan19/360LayoutAnalysis-general6-8n")


from latyas.layout.models.texteller.texteller_layout_model import TexTellerLayoutModel
model = TexTellerLayoutModel.from_pretrained("XiaHan19/texteller_rtdetr_r50vd_6x_coco")

# from latyas.ocr.models.easyocr.easyocr_ocr_config import EasyOCROCRConfig
# from latyas.ocr.models.easyocr.easyocr_ocr_model import EasyOCROCRModel
# ocr_model = EasyOCROCRModel(EasyOCROCRConfig())

from latyas.ocr.models.paddleocr.paddleocr_ocr_config import PaddleOCRConfig
from latyas.ocr.models.paddleocr.paddleocr_ocr_model import PaddleOCRModel
ocr_model = PaddleOCRModel(PaddleOCRConfig())


from latyas.layout.block import Block, BlockType
import pdf2image
import cv2
import numpy as np
import os
import tqdm

output_path = "./outputs/"

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)


file_path = "report5.pdf"
pages = pdf2image.convert_from_path(file_path)
for page_number, page in enumerate(tqdm.tqdm(pages)):
    page_img = np.asarray(page)
    page_shape = page_img.shape # (h, w, c)
    sf = (page_shape[0] // 256, page_shape[1] // 3)

    page_layout = model.detect(page_img)
    
    vis = page_layout.visualize()
    cv2.imwrite(os.path.join(output_path, f'output_{page_number}.jpg'), vis)

    # Sort Blocks
    sorted_bbox = []
    for bbox_i in range(len(page_layout)):
        x, y, x2, y2 = page_layout[bbox_i].shape.boundingbox
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
        sorted_bbox.append(((x//sf[1], y//sf[0], x2//sf[1], y2//sf[0]), bbox_i))
    sorted_bbox = sorted(sorted_bbox, key=lambda x: x[0])

    # OCR
    with open(os.path.join(output_path, f'output_{page_number}.txt'), "w", encoding="utf-8") as f:
        for k, bbox_i in sorted_bbox:
            block = page_layout[bbox_i]
            if block.kind in (BlockType.Text, BlockType.Title):
                text = ocr_model.detect(page_layout.crop_image(block))
                # text = text.replace("\n", "")
                f.write(text)
                f.write("\n\n")