import torch

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from typing import List
from typing import Tuple

from source.lm_classifier import device


def evaluate(model, test_iter: DataLoader, threshold=0.5) -> Tuple[List, List]:
    """
    Given a model and a test iterator
    returns the model prediction and the true labels

    y_pred :: (test_iter.shape[0])
    y_true :: (test_iter.shape[0])
    """
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():

        for batch in test_iter:

            batch = {k: v.to(device) for k, v in batch.items()}

            target = batch.pop('target')

            output = model(**batch)

            y_pred.extend(torch.argmax((output>threshold).float(), axis=-1).tolist())
            y_true.extend(target.tolist())

    return y_true, y_pred


def evaluate_probs(model, test_iter: DataLoader) -> Tuple[List, List]:
    """
    Given a model and a test iterator
    returns the model prediction and the true labels

    Returns the normalized probablitities

    y_pred :: (test_iter.shape[0])
    y_true :: (test_iter.shape[0])
    """
    y_pred = []
    y_true = []
    y_probs = []

    model.eval()
    sm = torch.nn.Softmax(dim=1)

    with torch.no_grad():

        for batch in test_iter:

            batch = {k: v.to(device) for k, v in batch.items()}

            target = batch.pop('target')

            output = model(**batch)

            y_pred.extend(torch.argmax(output, axis=-1).tolist())
            y_true.extend(target.tolist())
            y_probs.extend(sm(output).tolist())

    return y_true, y_pred, y_probs
