import pdf2image
import cv2
import numpy as np
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
general6_8n_pt = hf_hub_download(repo_id="qihoo360/360LayoutAnalysis", filename="general6-8n.pt")

# with blob.as_bytes_io() as file_path:
file_path = "report.pdf"
pages = pdf2image.convert_from_path(file_path)
for page_number, page in enumerate(pages):
    page_img = np.asarray(page)
    page_shape = page_img.shape # (h, w, c)

    model = YOLO(general6_8n_pt)
    results = model.predict(source=page)
    for result_i, result in enumerate(results):
        names = result.names
        # Detection
        result.boxes.xyxy  # box with xyxy format, (N, 4)
        result.boxes.xywh  # box with xywh format, (N, 4)
        result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
        result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
        result.boxes.conf  # confidence score, (N, 1)
        result.boxes.cls  # cls, (N, 1)

        # Sort by bbox
        sf = (page_shape[0] // 256, page_shape[1] // 3)
        sorted_bbox = []
        for bbox_i in range(result.boxes.xyxy.shape[0]):
            x, y, x2, y2 = result.boxes.xyxy[bbox_i, :].cpu().numpy()
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
            sorted_bbox.append(((x//sf[1], y//sf[0], x2//sf[1], y2//sf[0]), bbox_i))
        sorted_bbox = sorted(sorted_bbox, key=lambda x: x[0])
        # Crop Image
        for sorted_bbox_i, (_, bbox_i) in enumerate(sorted_bbox):
            x, y, x2, y2 = result.boxes.xyxy[bbox_i, :].cpu().numpy()
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
            bbox_image = page_img[y:y2, x:x2]
            bbox_cls = result.boxes.cls[bbox_i].item()

            bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join("outputs", f'output_with_labels_{page_number}_{result_i}_{sorted_bbox_i}_{names[bbox_cls]}.png'), bbox_image)
