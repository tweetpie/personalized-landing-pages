import torch
import logging

from source.lm_classifier import device

logger = logging.getLogger(__name__)


def save_checkpoint(path, model, valid_loss) -> None:
    """
    Writes the current state of `model` to `path`
    """
    logger.info(f"Saving checkpoint to '{path}'")
    torch.save({
        'model_state_dict': model.state_dict(),
        'valid_loss': valid_loss
    }, path)


def load_checkpoint(path, model) -> None:
    """
    Loads a model checkpoint from `path`
    """
    logger.info(f"Loading checkpoint from '{path}'")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']