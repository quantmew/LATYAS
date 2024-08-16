
import logging
from typing import List
from latyas.layout.layout import Layout

from transformers import AutoTokenizer, BertTokenizer, BertForNextSentencePrediction
import torch

from latyas.layout.reflow.position_based.position_reflow import position_sorting


def bert_sorting(page_layout: Layout, threshold=3) -> List[int]:
    position_blocks = position_sorting(page_layout)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForNextSentencePrediction.from_pretrained("bert-base-chinese")
    for bbox_i in range(len(position_blocks)):
        for bbox_j in range(bbox_i, len(position_blocks)):
            if bbox_i == bbox_j:
                continue
            lhs_bbox = page_layout[position_blocks[bbox_i]].shape.boundingbox
            rhs_bbox = page_layout[position_blocks[bbox_j]].shape.boundingbox
            
            if rhs_bbox[0] < lhs_bbox[2] and rhs_bbox[1] < lhs_bbox[3]:
                continue
            lhs_text = page_layout[position_blocks[bbox_i]].text
            rhs_text = page_layout[position_blocks[bbox_j]].text
            if lhs_text is None or rhs_text is None:
                continue
            # print("========")
            # print(lhs_text)
            # print("--------")
            # print(rhs_text)
            # print("========")
            
            # lhs_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lhs_text))
            # rhs_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(rhs_text))
            # input_ids = torch.tensor(tokenizer.build_inputs_with_special_tokens(lhs_ids, rhs_ids), dtype=torch.long).unsqueeze(0)
            # input_mask = torch.tensor(tokenizer.get_special_tokens_mask(lhs_ids, rhs_ids), dtype=torch.long).unsqueeze(0)
            # token_type = torch.tensor(tokenizer.create_token_type_ids_from_sequences(lhs_ids, rhs_ids), dtype=torch.long).unsqueeze(0)
            encoded = tokenizer.encode_plus(lhs_text, text_pair=rhs_text, return_tensors='pt')
            with torch.no_grad():
                # outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type)
                outputs = model(**encoded)
                logits = outputs.logits
                # print(logits[0, 0], logits[0, 1])
                if logits[0, 0] - logits[0, 1] > threshold:
                    old_ele = position_blocks[bbox_j]
                    del position_blocks[bbox_j]
                    position_blocks.insert(bbox_i + 1, old_ele)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.INFO)
    return position_blocks