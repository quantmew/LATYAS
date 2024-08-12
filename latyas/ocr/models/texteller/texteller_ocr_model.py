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
from typing import Optional, Union

import torch

from latyas.ocr.models.ocr_model import OCRModel
from latyas.ocr.ocr_utils import small_image_padding
from .texteller_ocr_config import TexTellerOCRConfig

from .ocr_model.model.TexTeller import TexTeller
from .ocr_model.utils.inference import inference as latex_inference
from .ocr_model.utils.to_katex import to_katex

class TexTellerOCRModel(OCRModel):
    def __init__(self, config: TexTellerOCRConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path
        
        self.latex_rec_model = TexTeller.from_pretrained()
        self.tokenizer = TexTeller.get_tokenizer()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> "TexTellerOCRModel":
        config = TexTellerOCRConfig.from_pretrained(pretrained_model_name_or_path)
        config._name_or_path = pretrained_model_name_or_path
        config._revision = revision
        return cls(config)

    def detect(self, image: Union["np.ndarray", "Image.Image"], num_beam=5) -> str:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = image
        if image_array.shape[0] < 400 or image_array.shape[1] < 400:
            image_array = small_image_padding(image_array)
        if torch.cuda.is_available():
            accelerator = "cuda"
        else:
            accelerator = "cpu"
        res = latex_inference(self.latex_rec_model, self.tokenizer, [image_array], accelerator=accelerator, num_beams=num_beam)
        res = to_katex(res[0])
        return res
