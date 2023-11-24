import logging

import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import PreTrainedTokenizer
from source.lm_classifier.data.text_dataset import TextDataset

from typing import Tuple
from typing import List

from source.lm_classifier.ops.sample import stratified_split

logger = logging.getLogger(__name__)


def create_datasets(tokenizer: PreTrainedTokenizer, filepath: str, split_ratios: List[float], stratify_by: str = 'label') -> Tuple[TextDataset, TextDataset, TextDataset]:
    """
    Creates a dataset from the file `filepath`
    Each record contains two fields: 'data', 'label'
    split_ratio :: [train, val, test]
    """
    logger.info(f"Creating a dataset from file in '{filepath}'")
    if type(filepath) == str:
        df = pd.read_csv(filepath)
    else:
        df = filepath

    train, val, test = stratified_split(df, stratify_by, split_ratios)

    dataset_train = TextDataset(train, tokenizer)
    dataset_val = TextDataset(val, tokenizer)
    dataset_test = TextDataset(test, tokenizer)

    logger.info(f"Training Set    '{len(dataset_train)}' examples")
    logger.info(f"Validation Set  '{len(dataset_val)}' examples")
    logger.info(f"Test Set        '{len(dataset_test)}' examples")

    return dataset_train, dataset_val, dataset_test


def create_testset(tokenizer: PreTrainedTokenizer, filepath: str) -> TextDataset:
    """
    Creates a dataset from the file `filepath`
    Each record contains two fields: 'data', 'label'
    """
    logger.info(f"Creating a dataset from file in '{filepath}'")
    if type(filepath) == str:
        df = pd.read_csv(filepath)
    else:
        df = filepath
    # df = df.sample(n=1000, random_state=42)

    dataset = TextDataset(df, tokenizer)

    logger.info(f"Test Set        '{len(dataset)}' examples")

    return dataset