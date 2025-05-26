# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html

import os
import torch
from torch.utils.data import Dataset

from byte_level_bpe import parse

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _encode(x, tokenizer):
    encoding = tokenizer(
        x,
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    if isinstance(x, str):
        return encoding["input_ids"][1: -1]
    else:
        return [token_ids[1: -1] for token_ids in encoding["input_ids"]]


class BookCorpusForRoBERTa(Dataset):
    def __init__(
        self,
        epubtxt_dir,
        tokenizer,
        seq_len,
        mode="full_sentences",
    ):
        self.epubtxt_dir = epubtxt_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mode = mode

        self.unk_id = tokenizer.unk_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.lines = parse(epubtxt_dir, with_document=True)

    def _to_bert_input(self, token_ids):
        # Add "[CLS]" and the first "[SEP]" tokens.
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        token_ids += [self.pad_id] * (self.seq_len - len(token_ids)) # Pad.
        return torch.as_tensor(token_ids)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        gt_token_ids = list()
        prev_doc = self.lines[idx][0]
        while True:
            if idx >= len(self.lines) - 1:
                break
                
            cur_doc, line = self.lines[idx]
            token_ids = _encode(line, tokenizer=self.tokenizer)
            if len(gt_token_ids) + len(token_ids) >= self.seq_len - 2:
                break

            if prev_doc != cur_doc:
                gt_token_ids.append(self.sep_id)

            gt_token_ids.extend(token_ids)
            prev_doc = cur_doc
            idx += 1

        gt_token_ids = self._to_bert_input(gt_token_ids)
        return gt_token_ids
