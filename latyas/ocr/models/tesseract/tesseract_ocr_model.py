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

import pytesseract

from latyas.ocr.models.ocr_model import OCRModel
from .tesseract_ocr_config import TesseractOCRConfig

class TesseractOCRModel(OCRModel):
    def __init__(self, config: TesseractOCRConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path
        self._tesseract_cmd = config.tesseract_cmd
        if os.path.exists(self._name_or_path):
            self.tesseract_cmd = os.path.join(self._name_or_path, self._tesseract_cmd)
        else:
            raise Exception("Cannot find the tesseract cmd file.")
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> 'TesseractOCRModel':
        config = TesseractOCRConfig.from_pretrained(pretrained_model_name_or_path)
        config._name_or_path = pretrained_model_name_or_path
        config._revision = revision
        return cls(config)

    def detect(self, image: Union["np.ndarray", "Image.Image"], threshold: float=0.3) -> str:
        text = pytesseract.image_to_string(image)
        return text