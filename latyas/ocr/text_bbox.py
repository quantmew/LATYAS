from typing import Optional

from latyas.layout.shape import Rectangle


class TextBoundingBox(object):
    def __init__(
        self,
        rect: Optional[Rectangle] = None,
        text: Optional[str] = None,
        confidence: float = 0.0,
    ) -> None:
        self._rect: Optional[Rectangle] = rect
        self._text: Optional[str] = text
        self._confidence: float = confidence

    @property
    def shape(self) -> Optional[Rectangle]:
        return self._rect

    @property
    def rect(self) -> Optional[Rectangle]:
        return self._rect

    @rect.setter
    def rect(self, new_rect: Optional[Rectangle]) -> None:
        self._rect = new_rect

    @property
    def text(self) -> Optional[str]:
        return self._text

    @text.setter
    def text(self, new_text: Optional[str]) -> None:
        self._text = new_text

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, new_confidence: float) -> None:
        self._confidence = new_confidence

    def __str__(self):
        return f"TextBoundingBox: {self._text}, Confidence: {self._confidence}"

    def __repr__(self):
        return f"TextBoundingBox(rect={self._rect}, text={self._text}, confidence={self._confidence})"