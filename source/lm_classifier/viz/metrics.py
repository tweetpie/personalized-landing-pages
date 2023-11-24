import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from source.lm_classifier.utils.metrics import load_metrics


logger = logging.getLogger(__name__)


def write_train_valid_loss(output_path: str) -> None:
    """
    Write the train and validation loss as a function of
    the global steps
    """
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(output_path + '/metric.pkl')

    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=14)

    png_path = os.path.join(output_path, 'train-valid-loss-vs-global-steps.png')
    logger.info(f"Saving train/validation loss Plot to '{png_path}'")

    plt.savefig(png_path)
