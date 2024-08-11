import json
import glob
import os
import tqdm

import cv2


from typing import List
from latyas.layout.block import Block, BlockType
from latyas.layout.models.ultralytics.ultralytics_layout_model import (
    UltralyticsLayoutModel,
)
work_path = "../MSDLA/dataset/zh/equity_research"
model = UltralyticsLayoutModel.from_pretrained("H:\\模型\\360LayoutAnalysis-report-8n")

# work_path = "../MSDLA/dataset/en/paper"
# model = UltralyticsLayoutModel.from_pretrained("H:\\模型\\360LayoutAnalysis-paper-8n")




jpg_files = glob.glob(os.path.join(work_path, "*.jpg"))
for file in tqdm.tqdm(list(jpg_files)):

    file_dir = os.path.dirname(file)
    file_base = os.path.basename(file)
    json_base = file_base[:-4] + ".json"
    json_path = os.path.join(file_dir, json_base)
    if os.path.exists(json_path):
        continue

    image = cv2.imread(file)
    page_layout = model.detect(image)

    label_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": file_base,
        "imageData": None,
        "imageHeight": image.shape[0],
        "imageWidth": image.shape[1],
    }

    for bbox_i in range(len(page_layout)):
        x, y, x2, y2 = page_layout[bbox_i].shape.boundingbox
        x, y, x2, y2 = float(x), float(y), float(x2), float(y2)
        shape = {
            "label": page_layout[bbox_i].kind.name,
            "points": [[x, y], [x2, y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
        }
        label_data["shapes"].append(shape)
    
    
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(label_data, ensure_ascii=False, indent=2))