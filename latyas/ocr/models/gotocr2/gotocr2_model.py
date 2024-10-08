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

from latyas.ocr.models.ocr_model import OCRModel
from latyas.ocr.ocr_utils import small_image_padding
from .gotocr2_config import GOTOCR2OCRConfig

from transformers import AutoModel, AutoTokenizer

class GOTOCR2OCRModel(OCRModel):
    def __init__(self, config: GOTOCR2OCRConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path

        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._name_or_path, revision=config._revision, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self._name_or_path,
            revision=config._revision,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> "GOTOCR2OCRModel":
        config = GOTOCR2OCRConfig.from_pretrained(pretrained_model_name_or_path)
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

        pil_image = Image.fromarray(image_array)
        res = self.model.chat(self.tokenizer, pil_image, ocr_type='ocr', gradio_input=True)
        return res
