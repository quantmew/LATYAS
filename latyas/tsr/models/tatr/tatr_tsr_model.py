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
from .tatr_tsr_config import TatrTSRConfig

class TatrTSRModel(OCRModel):
    def __init__(self, config: TatrTSRConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path
        
        pipe = TableExtractionPipeline(det_device=args.detection_device,
                            str_device=args.structure_device,
                            det_config_path=args.detection_config_path, 
                            det_model_path=args.detection_model_path,
                            str_config_path=args.structure_config_path, 
                            str_model_path=args.structure_model_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> "TatrTSRModel":
        config = TatrTSRConfig.from_pretrained(pretrained_model_name_or_path)
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
        
        


        return res
