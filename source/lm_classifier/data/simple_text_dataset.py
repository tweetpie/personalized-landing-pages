import torch
import pandas as pd

from pandas import DataFrame

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from typing import List

from source.lm_classifier.constants import MAX_SEQ_LEN


class SimpleTextDataset(Dataset):

    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

        self.encodings = tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LEN
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.texts)


