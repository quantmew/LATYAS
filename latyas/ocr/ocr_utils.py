
from typing import Tuple
import cv2
import numpy as np
from PIL import Image

def add_margin(image: np.ndarray, margin: int, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    elif isinstance(image, np.ndarray):
        image_array = image
    else:
        image_array = image

    height = image_array.shape[0]
    width = image_array.shape[1]

    bg_height = height + 2*margin
    bg_width = width + 2*margin
    background = np.ones((bg_height, bg_width, 3), np.uint8) * color

    # 计算将图像放置在背景中心的偏移量
    x_offset = margin
    y_offset = margin

    # 将原始图像放置在背景中心
    background[y_offset:y_offset + height, x_offset:x_offset + width] = image_array
    return background


def small_image_padding(image: np.ndarray, bg_size: int=800, bg_margin: int=160) -> np.ndarray:
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    elif isinstance(image, np.ndarray):
        image_array = image
    else:
        image_array = image

    image_array = cv2.resize(image_array, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    image_array = cv2.blur(image_array, (5, 5))
    image_array = add_margin(image_array, margin=bg_margin)
    height = image_array.shape[0]
    width = image_array.shape[1]

    bg_height = max(height, bg_size)
    bg_width = max(width, bg_size)
    background = np.ones((bg_height, bg_width, 3), np.uint8) * 255

    # 计算将图像放置在背景中心的偏移量
    x_offset = (bg_width - width) // 2
    y_offset = (bg_height - height) // 2

    # 将原始图像放置在背景中心
    background[y_offset:y_offset + height, x_offset:x_offset + width] = image_array
    return background
