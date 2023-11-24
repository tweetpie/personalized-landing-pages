import logging

from source.lm_classifier import device
from source.lm_classifier.constants import BATCH_SIZE

from torch.utils.data import DataLoader, Dataset

from typing import List
from typing import Tuple

logger = logging.getLogger(__name__)


def make_iterators(train: Dataset, val: Dataset, test: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    torch.utils.data.DataLoader is an iterator which provides several useful features

    * Batching the data
    * Shuffling the data
    * Load the data in parallel using multiprocessing workers.

    You can specify how exactly the samples need to be batched using collate_fn.
    However, default collate should work fine for most use cases.
    """
    logger.info(f"Creating DataLoaders for datasets")

    dataloader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_val, dataloader_test


def make_test_iterator(test: Dataset) -> DataLoader:
    """
    torch.utils.data.DataLoader is an iterator which provides several useful features

    * Batching the data
    * Shuffling the data
    * Load the data in parallel using multiprocessing workers.

    You can specify how exactly the samples need to be batched using collate_fn.
    However, default collate should work fine for most use cases.
    """
    logger.info(f"Creating DataLoader for test dataset")

    dataloader_test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return dataloader_test