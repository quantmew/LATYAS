from latyas.layout.block import BlockType
from latyas.layout.models.texteller.texteller_layout_config import TexTellerLayoutConfig
from latyas.layout.models.texteller.texteller_layout_model import TexTellerLayoutModel
from latyas.ocr.models.gotocr2.gotocr2_model import GOTOCR2OCRModel
from latyas.pipelines.base_pipeline import BasePipeline
from latyas.layout.models.ultralytics.ultralytics_layout_model import (
    UltralyticsLayoutModel,
)
from latyas.ocr.models.paddleocr.paddleocr_ocr_config import PaddleOCRConfig
from latyas.ocr.models.paddleocr.paddleocr_ocr_model import PaddleOCRModel

from latyas.tex_ocr.models.texteller.texteller_ocr_config import TexTellerTexOCRConfig
from latyas.tex_ocr.models.texteller.texteller_ocr_model import TexTellerEmbeddingTexOCRModel, TexTellerTexOCRModel
from latyas.tex_ocr.models.texmix.texmix_model import TexMixMixTexOCRModel
from latyas.tex_ocr.models.texmix.texmix_config import TexMixMixTexOCRConfig
from latyas.tsr.models.gotocr2.gotocr2_model import GOTOCR2TSRModel

class PaperPipeline(BasePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.add_layout_model(
            "layout_360general",
            UltralyticsLayoutModel.from_pretrained(
                "XiaHan19/360LayoutAnalysis-paper-8n"
            ),
        )

        self.add_layout_model(
            "layout_texteller",
            TexTellerLayoutModel.from_pretrained(
                "XiaHan19/texteller_rtdetr_r50vd_6x_coco"
            ),
        )

        text_model = PaddleOCRModel(PaddleOCRConfig(lang="en"))
        llm_text_model = GOTOCR2OCRModel.from_pretrained('stepfun-ai/GOT-OCR2_0', revision="cf6b7386bc89a54f09785612ba74cb12de6fa17c")
        tex_model = TexTellerTexOCRModel.from_pretrained("OleehyO/TexTeller")
        embed_tex_model = TexTellerEmbeddingTexOCRModel.from_pretrained("OleehyO/TexTeller")
        self.add_ocr_model("ocr_paddle", llm_text_model)
        self.add_ocr_model("ocr_texteller", tex_model)
        self.add_ocr_model(
            "ocr_texmix", TexMixMixTexOCRModel(embed_tex_model, text_model, TexMixMixTexOCRConfig())
        )
        table_model = GOTOCR2TSRModel.from_pretrained('stepfun-ai/GOT-OCR2_0', revision="cf6b7386bc89a54f09785612ba74cb12de6fa17c")
        self.add_ocr_model("tsr_gotocr2", table_model)
        
        self.add_ocr_rule(BlockType.Title, "ocr_paddle")
        self.add_ocr_rule(BlockType.Text, "ocr_paddle")
        self.add_ocr_rule(BlockType.Caption, "ocr_paddle")
        self.add_ocr_rule(BlockType.TableCaption, "ocr_paddle")
        self.add_ocr_rule(BlockType.FigureCaption, "ocr_paddle")
        self.add_ocr_rule(BlockType.Reference, "ocr_paddle")
        self.add_ocr_rule(BlockType.Header, "ocr_paddle")
        self.add_ocr_rule(BlockType.Footer, "ocr_paddle")
        
        self.add_ocr_rule(BlockType.Equation, "ocr_texteller")
        self.add_ocr_rule(BlockType.EmbedEq, "ocr_texteller")
        self.add_ocr_rule(BlockType.TextWithEquation, "ocr_texmix")
        self.add_ocr_rule(BlockType.Table, "tsr_gotocr2")
        
        
