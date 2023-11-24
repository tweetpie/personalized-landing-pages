import logging
import torch

from source.lm_classifier import device


logger = logging.getLogger(__name__)


def save_metrics(path, train_loss_list, valid_loss_list, global_steps_list) -> None:
    """
    Saves metrics in `path`
    """
    logger.info(f"Saving metrics to '{path}'")
    state_dict = {
        'train_loss_list': train_loss_list,
        'valid_loss_list': valid_loss_list,
        'global_steps_list': global_steps_list
    }
    torch.save(state_dict, path)


def load_metrics(path) -> None:
    """
    Loads metrics from `path`
    """
    logger.info(f"Loading metrics from '{path}'")
    state_dict = torch.load(path, map_location=device)
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

