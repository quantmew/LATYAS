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
from .texteller_ocr_config import TexTellerTexOCRConfig, TexTellerEmbeddingTexOCRConfig

from .ocr_model.model.TexTeller import TexTeller
from .ocr_model.utils.inference import inference as latex_inference
from .ocr_model.utils.to_katex import to_katex


class TexTellerTexOCRModel(TexOCRModel):
    def __init__(self, config: TexTellerTexOCRConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path

        self.latex_detect_model = TexTellerLayoutModel.from_pretrained(
            "XiaHan19/texteller_rtdetr_r50vd_6x_coco"
        )
        self.latex_rec_model = TexTeller.from_pretrained()
        self.tokenizer = TexTeller.get_tokenizer()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> "TexTellerTexOCRModel":
        config = TexTellerTexOCRConfig.from_pretrained(pretrained_model_name_or_path)
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

        infer_config = PredictConfig(self.latex_detect_model.cfg_path)
        latex_det_model = InferenceSession(self.latex_detect_model.weights_path)
        latex_bboxes = latex_det_predict(image_array, latex_det_model, infer_config)
        
        bboxs = []
        for bbox in latex_bboxes:
            x, y = bbox.p.x, bbox.p.y
            x2, y2 = x + bbox.w, y + bbox.h
            if bbox.label == "isolated":
                block_type = BlockType.Equation
            else:
                block_type = BlockType.EmbedEq
            if block_type == BlockType.Equation:
                bboxs.append(
                    TextBoundingBox(Rectangle(x, y, x2, y2), bbox.content, bbox.confidence),
                )

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
        if torch.cuda.is_available():
            accelerator = "cuda"
        else:
            accelerator = "cpu"
        res = latex_inference(
            self.latex_rec_model,
            self.tokenizer,
            [image_array],
            accelerator=accelerator,
            num_beams=num_beam,
        )
        res = to_katex(res[0])
        return res


class TexTellerEmbeddingTexOCRModel(EmbeddingTexOCRModel):
    def __init__(self, config: TexTellerEmbeddingTexOCRConfig) -> None:
        self.config = config
        self._name_or_path = config._name_or_path

        self.latex_detect_model = TexTellerLayoutModel.from_pretrained(
            "XiaHan19/texteller_rtdetr_r50vd_6x_coco"
        )
        self.latex_rec_model = TexTeller.from_pretrained()
        self.tokenizer = TexTeller.get_tokenizer()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        revision: str = "main",
        **kwargs,
    ) -> "TexTellerEmbeddingTexOCRModel":
        config = TexTellerEmbeddingTexOCRConfig.from_pretrained(pretrained_model_name_or_path)
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

        infer_config = PredictConfig(self.latex_detect_model.cfg_path)
        latex_det_model = InferenceSession(self.latex_detect_model.weights_path)
        latex_bboxes = latex_det_predict(image_array, latex_det_model, infer_config)
        
        bboxs = []
        for bbox in latex_bboxes:
            x, y = bbox.p.x, bbox.p.y
            x2, y2 = x + bbox.w, y + bbox.h
            if bbox.label == "isolated":
                block_type = BlockType.Equation
            else:
                block_type = BlockType.EmbedEq
            if block_type == BlockType.EmbedEq:
                bboxs.append(
                    TextBoundingBox(Rectangle(x, y, x2, y2), bbox.content, bbox.confidence),
                )

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
        if torch.cuda.is_available():
            accelerator = "cuda"
        else:
            accelerator = "cpu"
        res = latex_inference(
            self.latex_rec_model,
            self.tokenizer,
            [image_array],
            accelerator=accelerator,
            num_beams=num_beam,
        )
        res = to_katex(res[0])
        return res
