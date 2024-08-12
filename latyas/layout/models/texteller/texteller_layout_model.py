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
from latyas.layout.block import Block, BlockType
from latyas.layout.layout import Layout
from latyas.layout.models.layout_config import LayoutConfig
from latyas.layout.models.layout_model import LayoutModel

from onnxruntime import InferenceSession
from .det_model.inference import PredictConfig
from .thrid_party.paddleocr.infer import predict_det, predict_rec
from .det_model.inference import predict as latex_det_predict

from huggingface_hub import hf_hub_download

from latyas.layout.shape import Rectangle

from .texteller_layout_config import TexTellerLayoutConfig

class TexTellerLayoutModel(LayoutModel):
    def __init__(self, config: TexTellerLayoutConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path
        self._cfg_name = config.cfg_name
        self._weights_name = config.weights_name
        if os.path.exists(self._name_or_path):
            self.cfg_path = os.path.join(self._name_or_path, self._cfg_name)
        else:
            self.cfg_path = hf_hub_download(repo_id=self._name_or_path, filename=self._cfg_name)
        
        if os.path.exists(self._name_or_path):
            self.weights_path = os.path.join(self._name_or_path, self._weights_name)
        else:
            self.weights_path = hf_hub_download(repo_id=self._name_or_path, filename=self._weights_name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> 'TexTellerLayoutModel':
        config = TexTellerLayoutConfig.from_pretrained(pretrained_model_name_or_path)
        config._name_or_path = pretrained_model_name_or_path
        config._revision = revision
        return cls(config)

    def detect(self, image: Union["np.ndarray", "Image.Image"], threshold: float=0.3) -> Layout:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image

        page_layout = Layout(page=image_array)
        infer_config = PredictConfig(self.cfg_path)
        latex_det_model = InferenceSession(self.weights_path)
        latex_bboxes = latex_det_predict(image_array, latex_det_model, infer_config)
        
        for bbox in latex_bboxes:
            x, y = bbox.p.x, bbox.p.y
            x2, y2 = x + bbox.w, y + bbox.h
            if bbox.label == "isolated":
                block_type = BlockType.Equation
            else:
                block_type = BlockType.EmbedEq
            page_layout.insert(
                0,
                Block(Rectangle(x, y, x2, y2), block_type),
            )

        page_layout.page_sort()
        page_layout.remove_overlapping()
        return page_layout
