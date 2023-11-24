import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import f1_score

from typing import List


logger = logging.getLogger(__name__)


def classification_report(y_true: List, y_pred: List) -> None:
    """
    Calculates and logs the classification report
    """
    report = sk_classification_report(y_true, y_pred, digits=4)
    logger.info("Classification Report")
    logger.info(report)


def weighted_f1_score(y_true: List, y_pred: List) -> float:
    """
    Calculate metrics for each label, and find their average weighted by support
    (the number of true instances for each label).
    This alters ‘macro’ to account for label imbalance;
    it can result in an F-score that is not between precision and recall.
    """
    score = f1_score(y_true, y_pred, average='weighted')
    return score


def write_confusion_matrix(y_true: List, y_pred: List, output_path: str) -> None:
    """
    Creates and writes confusion matrix plot to `output_path`
    """
    cm = confusion_matrix(y_true, y_pred)

    ax = plt.subplot()

    try:
        ax.get_legend().remove()
    except:
        pass

    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    png_path = os.path.join(output_path, 'confusion-matrix.png')
    logger.info(f"Saving Confusion Matrix Plot to '{png_path}'")
    plt.savefig(png_path)
