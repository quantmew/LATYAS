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
import numpy as np
from PIL import Image
from typing import Optional, Union

import easyocr

from latyas.ocr.models.ocr_model import OCRModel
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

    def detect(self, image: Union["np.ndarray", "Image.Image"]) -> str:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
        result = self.model.readtext(
            image_array, decoder="beamsearch", beamWidth=5, paragraph=True, detail=0
        )
        return "\n".join(result)
