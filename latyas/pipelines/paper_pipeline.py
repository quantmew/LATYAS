from latyas.layout.block import BlockType
from latyas.layout.models.texteller.texteller_layout_config import TexTellerLayoutConfig
from latyas.layout.models.texteller.texteller_layout_model import TexTellerLayoutModel
from latyas.pipelines.base_pipeline import BasePipeline
from latyas.layout.models.ultralytics.ultralytics_layout_model import (
    UltralyticsLayoutModel,
)
from latyas.ocr.models.paddleocr.paddleocr_ocr_config import PaddleOCRConfig
from latyas.ocr.models.paddleocr.paddleocr_ocr_model import PaddleOCRModel

from latyas.ocr.models.texteller.texteller_ocr_config import TexTellerOCRConfig
from latyas.ocr.models.texteller.texteller_ocr_model import TexTellerOCRModel


class PaperPipeline(BasePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.add_layout_model(
            "layout_360general",
            UltralyticsLayoutModel.from_pretrained(
                "XiaHan19/360LayoutAnalysis-general6-8n"
            ),
        )

        self.add_layout_model(
            "layout_texteller",
            TexTellerLayoutModel.from_pretrained(
                "XiaHan19/texteller_rtdetr_r50vd_6x_coco"
            ),
        )

        self.add_ocr_model("ocr_paddle", PaddleOCRModel(PaddleOCRConfig(lang="en")))
        self.add_ocr_model(
            "ocr_texteller", TexTellerOCRModel.from_pretrained("OleehyO/TexTeller")
        )

        self.add_ocr_rule(BlockType.Title, "ocr_paddle")
        self.add_ocr_rule(BlockType.Text, "ocr_paddle")
        self.add_ocr_rule(BlockType.Caption, "ocr_paddle")
        self.add_ocr_rule(BlockType.TableCaption, "ocr_paddle")
        self.add_ocr_rule(BlockType.FigureCaption, "ocr_paddle")
        self.add_ocr_rule(BlockType.Equation, "ocr_texteller")
        self.add_ocr_rule(BlockType.EmbedEq, "ocr_texteller")
