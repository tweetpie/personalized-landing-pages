import torch
import pandas as pd

from pandas import DataFrame

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from source.lm_classifier.constants import MAX_SEQ_LEN


class TextDataset(Dataset):

    def __init__(self, df: DataFrame, tokenizer: PreTrainedTokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.labels = list(df['label'])

        self.encodings = tokenizer(
            list(df['text']),
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LEN
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['target'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.df.shape[0]


