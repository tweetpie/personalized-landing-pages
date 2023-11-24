import logging
import torch

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from typing import List
from typing import Tuple

from source.lm_classifier import device
from source.lm_classifier.constants import BATCH_SIZE
from source.lm_classifier.data.simple_text_dataset import SimpleTextDataset

logger = logging.getLogger(__name__)


def predict(model, iterator: DataLoader) -> Tuple[List, List]:
    """
    Given a model and a data iterator
    returns the model prediction

    y_pred :: (iter.shape[0])
    """
    y_pred = []
    y_probs = []

    model.eval()

    sm = torch.nn.Softmax(dim=1)

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            logger.info(f"Batch '{i}/{len(iterator)}'")

            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)

            y_pred.extend(torch.argmax(output, axis=-1).tolist())
            y_probs.extend(sm(output).tolist())

    return y_pred, y_probs


def predict_texts(model, tokenizer, texts: List[str]) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Returns a model inference for each text entry in `texts`

    * List with the index of the predicted class
    * List with the normalized of all classes (normalized)
    """
    logger.info(f"Creating a iterator with batch size '{BATCH_SIZE}'")
    dataset = SimpleTextDataset(texts, tokenizer)
    iterator = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info(f"Number of batches '{len(iterator)}'")

    y_pred, y_probs = predict(model, iterator)

    return y_pred, y_probs


