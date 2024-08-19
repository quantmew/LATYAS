# coding=utf-8
# Copyright 2024 The LATYAS team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Union

import torch
from onnxruntime import InferenceSession

from latyas.layout.block import BlockType
from latyas.layout.layout import Layout
from latyas.layout.models.texteller.det_model.inference import PredictConfig
from latyas.layout.models.texteller.det_model.inference import predict as latex_det_predict

from latyas.layout.models.texteller.texteller_layout_model import TexTellerLayoutModel
from latyas.layout.reflow.position_based.xy_cut_reflow import xy_cut_reflow
from latyas.layout.shape import Rectangle
from latyas.ocr.models.ocr_model import OCRModel
from latyas.ocr.models.texmix.texmix_config import TexMixMixTexOCRConfig
from latyas.ocr.models.texocr_model import EmbeddingTexOCRModel, MixTexOCRModel, TexOCRModel
from latyas.ocr.ocr_utils import small_image_padding
from latyas.ocr.text_bbox import TextBoundingBox

# counter = 0

class TexMixMixTexOCRModel(MixTexOCRModel):
    def __init__(self, tex_model: EmbeddingTexOCRModel, text_model: OCRModel, config: TexMixMixTexOCRConfig) -> None:
        self.tex_model = tex_model
        self.text_model = text_model 
        self.config = config
        self._name_or_path = config._name_or_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> "TexMixMixTexOCRModel":
        config = TexMixMixTexOCRConfig.from_pretrained(pretrained_model_name_or_path)
        config._name_or_path = pretrained_model_name_or_path
        config._revision = revision
        return cls(config)

    def detect(
        self, image: Union["np.ndarray", "Image.Image"]
    ) -> List[TextBoundingBox]:

        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image

        raise NotImplementedError("The detect method of TexMixMixTexOCRModel has not been implemented")

    def recognize(self, image: Union["np.ndarray", "Image.Image"]) -> str:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = image

        if image_array.shape[0] < 400 or image_array.shape[1] < 400:
            image_array = small_image_padding(image_array, blur=0)

        snippet_bboxs: List[TextBoundingBox] = []
        equation_bboxs = self.tex_model.detect(image_array)

        local_layout = Layout()
        local_layout._page = image_array
        masked_layout = local_layout.copy()

        # Get bbox from embedding equations
        for eq_bbox in equation_bboxs:
            text = self.tex_model.recognize(local_layout.crop_image(eq_bbox))
            bbox = TextBoundingBox(eq_bbox.shape, "$" + text + "$", eq_bbox.confidence)
            snippet_bboxs.append(bbox)
            masked_layout.mask_image(eq_bbox)

        # Sort bboxs
        sorted_bbox = [(e.shape.boundingbox[0], e) for e in equation_bboxs]
        sorted_bbox = sorted(sorted_bbox, key=lambda x: x[0])
        equation_bboxs = [e[1] for e in sorted_bbox]

        # Get bbox from text
        text_bboxs = self.text_model.detect(masked_layout._page)

        # Reflow bboxs
        reflow_indices = xy_cut_reflow(text_bboxs, margin=0, horizontal_first=False)
        text_bboxs = [text_bboxs[idx] for idx in reflow_indices]

        # Split text bbox
        split_text_bboxs: List[TextBoundingBox] = []
        for bbox in text_bboxs:
            need_split_eq_blocks: List[TextBoundingBox] = []
            for eq_bbox in equation_bboxs:
                if eq_bbox.shape.intersect(bbox.shape).area / eq_bbox.shape.area > 0.8:
                    need_split_eq_blocks.append(eq_bbox)
            cur_rect = bbox.shape
            for eq_block in need_split_eq_blocks:
                boundingbox = eq_block.shape.boundingbox
                lhs_text_bbox, _ = cur_rect.split_x(boundingbox[0])
                _, rhs_text_bbox = cur_rect.split_x(boundingbox[2])
                if lhs_text_bbox is not None:
                    split_text_bboxs.append(TextBoundingBox(rect=lhs_text_bbox, confidence=1.0))
                cur_rect = rhs_text_bbox
            if cur_rect is not None:
                split_text_bboxs.append(TextBoundingBox(rect=cur_rect, confidence=1.0))
        snippet_bboxs.extend(split_text_bboxs)

        # Rerecognize
        for bbox in snippet_bboxs:
            if bbox.text is not None:
                continue
            text = self.text_model.recognize(local_layout.crop_image(bbox))
            bbox.text = text

        # global counter
        # vis = local_layout.draw_bboxs([box.shape for box in snippet_bboxs])
        # vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f"./outputs/output_{counter}.jpg", vis)
        # counter += 1

        # Reflow bboxs
        shrinked_snippet_bboxs = [
            TextBoundingBox(rect=e.shape.shrink(0.5)) for e in snippet_bboxs
        ]
        reflow_indices = xy_cut_reflow(
            shrinked_snippet_bboxs, margin=0, horizontal_first=False
        )
        snippet_bboxs = [snippet_bboxs[idx] for idx in reflow_indices]

        text_list = []
        for bbox in snippet_bboxs:
            text_list.append(bbox.text)
        return " ".join(text_list)
