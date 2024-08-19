import argparse
import pypdfium2
import cv2
import numpy as np
import os
import tqdm

from latyas.layout.block import BlockType
from latyas.pipelines.book_pipeline import BookPipeline
from latyas.pipelines.paper_pipeline import PaperPipeline
from latyas.pipelines.report_pipeline import ReportPipeline


def pdf2text(pdf_path: str, mode: str):
    mode = mode.lower()
    if mode == "report":
        pipeline = ReportPipeline()
    elif mode == "paper":
        pipeline = PaperPipeline()
    elif mode == "book":
        pipeline = BookPipeline()
    else:
        raise Exception("Unsupported mode.")
    pdf_reader = pypdfium2.PdfDocument(pdf_path, autoclose=True)
    texts = []
    try:
        for page_number, page in tqdm.tqdm(enumerate(pdf_reader), total=len(pdf_reader)):
            page_layout = pipeline.analyze_pdf(page)
            text = []
            for i in range(len(page_layout)):
                block = page_layout[i]
                if block._text is None:
                    continue
                if block.kind == BlockType.EmbedEq:
                    continue
                elif block.kind == BlockType.Equation:
                    text.append("\n$$\n" + block._text + "\n$$\n")
                else:
                    text.append(block._text)
            texts.append(text)
            page.close()
            # break
    finally:
        pdf_reader.close()
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF file.")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file") # default="report6.pdf"
    parser.add_argument("--out", type=str, help="Path to the text file", default="out.txt")
    parser.add_argument("--mode", type=str, help="Parse mode", default="paper")

    args = parser.parse_args()

    pdf_path = args.pdf
    print(f"PDF file path: {pdf_path}")

    texts = pdf2text(pdf_path, mode=args.mode)
    with open(args.out, "w", encoding="utf-8") as f:
        for text in texts:
            f.write("\n\n\n".join(text))
