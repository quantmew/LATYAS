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

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from latyas.layout.shape import Rectangle

from .ultralytics_layout_config import UltralyticsLayoutConfig

class UltralyticsLayoutModel(LayoutModel):
    def __init__(self, config: UltralyticsLayoutConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path
        self._checkpoint_path = config.checkpoint_path
        if os.path.exists(self._name_or_path):
            self.ckpt_path = os.path.join(self._name_or_path, self._checkpoint_path)
        else:
            self.ckpt_path = hf_hub_download(repo_id=self._name_or_path, filename=self._checkpoint_path)
        self.model = YOLO(self.ckpt_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> 'UltralyticsLayoutModel':
        config = UltralyticsLayoutConfig.from_pretrained(pretrained_model_name_or_path)
        config._name_or_path = pretrained_model_name_or_path
        config._revision = revision
        return cls(config)

    def detect(self, image: Union["np.ndarray", "Image.Image"], threshold: float=0.3) -> Layout:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image

        page_layout = Layout(page=image_array)

        results = self.model.predict(source=image_array)
        if len(results) != 1:
            raise Exception("The number of prediction results is not one.")
        result = results[0]

        # Detection
        names = result.names
        result.boxes.xyxy  # box with xyxy format, (N, 4)
        result.boxes.xywh  # box with xywh format, (N, 4)
        result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
        result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
        result.boxes.conf  # confidence score, (N, 1)
        result.boxes.cls  # cls, (N, 1)

        for bbox_i in range(result.boxes.xyxy.shape[0]):
            label_id = result.boxes.cls[bbox_i].item()
            conf = result.boxes.conf[bbox_i].item()
            x, y, x2, y2 = result.boxes.xyxy[bbox_i, :].cpu().numpy()
            x, y, x2, y2 = float(x), float(y), float(x2), float(y2)
            if conf > threshold:
                page_layout.insert(
                    0,
                    Block(Rectangle(x, y, x2, y2), BlockType.from_str(names[label_id])),
                )
        # page_layout.page_sort()
        page_layout.remove_overlapping()
        return page_layout
