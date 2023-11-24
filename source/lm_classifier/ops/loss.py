import logging
import torch

from source.lm_classifier import device
from typing import List


logger = logging.getLogger(__name__)


def cross_entropy_loss(weights: List[float] = None):
    """
    Cross entropy loss
    weight_label_i = total_number_of_samples / nr_samples_of_label_i
    """
    logger.info(f"Creating Cross Entropy Loss with weights '{weights}''")
    if weights is not None:
        weight = torch.tensor(weights).to(device)
        loss = torch.nn.CrossEntropyLoss(weight)
    else:
        loss = torch.nn.CrossEntropyLoss()

    return loss


def bce_with_logits_loss(pos_weight: float):
    """
    Loss for binary classification problems, if we have two classes respectively
    `pos_weight` takes as input the positive class weight

    e.g.
        0 900
        1 100
        (in this case 900/100 = 9) -> weight = 9.0

    For a binary classification you could use:

    * one output value and use nn.BCEWithLogitsLoss
    * or two outputs and use nn.CrossentropyLoss

    Both approaches would work, but are a bit different in their implementation.
    While the former is used for a classical binary classification as well as a multi-label classification
    (each sample might belong to zero, one, or multiple classes),
    the latter is used for a multi-class classification (each sample belongs to one class only).
    """
    class_weight = torch.FloatTensor([pos_weight]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight)
    return criterion