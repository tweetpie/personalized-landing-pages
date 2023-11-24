import os
from source.lm_classifier import device
from source.lm_classifier.utils.checkpoints import load_checkpoint


def load_model(model, output_path: str):
    """
    Loads the model in `output_path` and returns it
    in the correct device
    """
    model = model.to(device)
    load_checkpoint(os.path.join(output_path, 'model.pkl'), model)
    return model

