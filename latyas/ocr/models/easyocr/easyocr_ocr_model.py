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

import easyocr

from latyas.layout.shape import Rectangle
from latyas.ocr.models.ocr_model import OCRModel
from latyas.ocr.ocr_utils import small_image_padding
from latyas.ocr.text_bbox import TextBoundingBox
from .easyocr_ocr_config import EasyOCROCRConfig


class EasyOCROCRModel(OCRModel):
    def __init__(self, config: EasyOCROCRConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path
        self._lang_list = config.lang_list

        self.model = easyocr.Reader(self._lang_list)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> "EasyOCROCRModel":
        config = EasyOCROCRConfig.from_pretrained(pretrained_model_name_or_path)
        config._name_or_path = pretrained_model_name_or_path
        config._revision = revision
        return cls(config)

    def recognize(self, image: Union["np.ndarray", "Image.Image"]) -> str:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = image
        # 如果图像的宽度或高度小于400，则放置在800x800的白色背景上
        if image_array.shape[0] < 400 or image_array.shape[1] < 400:
            image_array = small_image_padding(image_array)
        results = self.model.readtext(
            image_array, decoder="beamsearch", beamWidth=5, paragraph=True, detail=0
        )

        return "\n".join(results)

    def detect(
        self, image: Union["np.ndarray", "Image.Image"]
    ) -> List[TextBoundingBox]:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = image
        results = self.model.readtext(
            image_array, decoder="beamsearch", beamWidth=5, paragraph=False, detail=1
        )
        ret = []
        for result in results:
            point0, point1, point2, point3 = result[0]
            text = result[1]
            confidence = result[2]
            bbox = TextBoundingBox()
            bbox.rect = Rectangle(
                x_1=min(point0[0], point1[0], point2[0], point3[0]),
                y_1=min(point0[1], point1[1], point2[1], point3[1]),
                x_2=max(point0[0], point1[0], point2[0], point3[0]),
                y_2=max(point0[1], point1[1], point2[1], point3[1]),
            )
            bbox.text = text
            bbox.confidence = confidence
            ret.append(bbox)
        return ret
