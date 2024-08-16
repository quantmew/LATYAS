from latyas.layout.block import BlockType
from latyas.pipelines.base_pipeline import BasePipeline
from latyas.layout.models.ultralytics.ultralytics_layout_model import (
    UltralyticsLayoutModel,
)
from latyas.ocr.models.paddleocr.paddleocr_ocr_config import PaddleOCRConfig
from latyas.ocr.models.paddleocr.paddleocr_ocr_model import PaddleOCRModel

from latyas.ocr.models.texteller.texteller_ocr_config import TexTellerOCRConfig
from latyas.ocr.models.texteller.texteller_ocr_model import TexTellerOCRModel

class ReportPipeline(BasePipeline):
    def __init__(self) -> None:
        self.add_layout_model("layout_360general", UltralyticsLayoutModel.from_pretrained(
            "XiaHan19/360LayoutAnalysis-general6-8n"
        ))
        
        self.add_ocr_model("ocr_paddle", PaddleOCRModel(PaddleOCRConfig()))

        self.add_ocr_rule(BlockType.Title, "ocr_paddle")
        self.add_ocr_rule(BlockType.Text, "ocr_paddle")
        self.add_ocr_rule(BlockType.Caption, "ocr_paddle")

