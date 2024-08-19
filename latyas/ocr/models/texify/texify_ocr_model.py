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
from latyas.layout.models.texteller.det_model.inference import PredictConfig
from latyas.layout.models.texteller.det_model.inference import predict as latex_det_predict

from latyas.layout.models.texteller.texteller_layout_model import TexTellerLayoutModel
from latyas.layout.shape import Rectangle
from latyas.ocr.models.ocr_model import OCRModel
from latyas.ocr.models.texocr_model import EmbeddingTexOCRModel, TexOCRModel
from latyas.ocr.ocr_utils import small_image_padding
from latyas.ocr.text_bbox import TextBoundingBox
from .texify_ocr_config import TexifyTexOCRConfig

class TexifyTexOCRModel(TexOCRModel):
    def __init__(self, config: TexifyTexOCRConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> "TexifyTexOCRModel":
        config = TexifyTexOCRConfig.from_pretrained(pretrained_model_name_or_path)
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

        return bboxs

    def recognize(self, image: Union["np.ndarray", "Image.Image"], num_beam=5) -> str:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = image

        if image_array.shape[0] < 400 or image_array.shape[1] < 400:
            image_array = small_image_padding(image_array)
        pass
        return res
