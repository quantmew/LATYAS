

from typing import Optional, Union
from latyas.layout.shape import Shape


from enum import Enum

class BlockType(Enum):
    Unknown = 0
    Text = 1
    Title = 2
    Caption = 3
    Figure = 4
    FigureCaption = 5
    Table = 6
    TableCaption = 7
    Header = 8
    Footer = 9
    Reference = 10
    Equation = 11
    EmbedEq = 12
    TOC = 13
    List = 14
    Icon = 15
    QRCode = 16
    BarCode = 17

    @classmethod
    def from_str(cls, s: str) -> 'BlockType':
        if 'text' in s.lower():
            return BlockType.Text
        elif 'title' in s.lower():
            return BlockType.Title
        elif 'caption' in s.lower() and 'figure' in s.lower():
            return BlockType.FigureCaption
        elif 'caption' in s.lower() and 'table' in s.lower():
            return BlockType.TableCaption
        elif 'caption' in s.lower():
            return BlockType.Caption
        elif 'figure' in s.lower():
            return BlockType.Figure
        elif 'table' in s.lower():
            return BlockType.Table
        elif 'header' in s.lower():
            return BlockType.Header
        elif 'footer' in s.lower():
            return BlockType.Footer
        elif 'reference' in s.lower():
            return BlockType.Reference
        elif 'embedeq' in s.lower():
            return BlockType.EmbedEq
        elif 'equation' in s.lower():
            return BlockType.Equation
        elif 'toc' in s.lower():
            return BlockType.TOC
        elif 'list' in s.lower():
            return BlockType.List
        elif 'icon' in s.lower():
            return BlockType.Icon
        elif 'qrcode' in s.lower():
            return BlockType.QRCode
        elif 'barcode' in s.lower():
            return BlockType.BarCode
        else:
            return BlockType.Unknown


# 定义一个颜色映射字典，值为 RGB 元组
BLOCK_TYPE_COLOR_MAP = {
    BlockType.Text: (255, 0, 0),         # 红色
    BlockType.Title: (0, 0, 255),        # 蓝色
    BlockType.Caption: (0, 255, 0),      # 绿色
    BlockType.Figure: (255, 165, 0),     # 橙色
    BlockType.FigureCaption: (128, 0, 128),  # 紫色
    BlockType.Table: (255, 255, 0),      # 黄色
    BlockType.TableCaption: (0, 255, 255),  # 青色
    BlockType.Header: (255, 0, 255),     # 品红
    BlockType.Footer: (165, 42, 42),     # 棕色
    BlockType.Reference: (255, 192, 203), # 粉色
    BlockType.Equation: (128, 128, 128), # 灰色
    BlockType.EmbedEq: (100, 100, 100), # 灰色
    BlockType.TOC: (0, 128, 128),        # 水鸭绿
    BlockType.List: (128, 128, 0),        # 橄榄色
    BlockType.Icon: (0, 0, 0), # 黑色
    BlockType.QRCode: (0, 128, 0), # 绿色
    BlockType.BarCode: (128, 0, 0), # 深红色
    BlockType.Unknown: (192, 192, 192), # 浅灰色
}

class Block(object):
    def __init__(self, shape: Shape, kind: BlockType=BlockType.Unknown) -> None:
        self._shape = shape
        self._kind = kind
        self._text: Optional[str] = None
    
    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def kind(self) -> BlockType:
        return self._kind
    
    @property
    def text(self) -> Optional[str]:
        return self._text
    
    def set_text(self, text: str):
        self._text = text