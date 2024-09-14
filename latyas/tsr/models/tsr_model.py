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

import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from typing import Union

from latyas.models.latyas_model import LatyasModel


class TSRModel(LatyasModel):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def recognize(self, image: Union["np.ndarray", "Image.Image"]) -> str:
        pass
