import os
import sys

import torch
import json
import logging
import pandas as pd

from torch.utils.data import DataLoader
from torchviz import make_dot

from source.lm_classifier import device
from source.lm_classifier.io.model import load_model

from source.lm_classifier.ops.evaluation import evaluate
from source.lm_classifier.ops.loss import cross_entropy_loss
from source.lm_classifier.ops.metrics import classification_report, write_confusion_matrix
from source.lm_classifier.ops.weights import calculate_multiclass_weights

from sklearn.metrics import f1_score

from source.lm_classifier.processes.pretrain import pretrain
from source.lm_classifier.processes.train import train

from source.lm_classifier.models import make_model
from source.lm_classifier.tokenizer import create_tokenizer

from source.lm_classifier.utils.directories import make_run_dir
from source.lm_classifier.viz.metrics import write_train_valid_loss

from source.lm_classifier.data.dataset import create_datasets, create_testset
from source.lm_classifier.data.iterator import make_iterators, make_test_iterator

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Dict
from typing import Tuple
from typing import List

logger = logging.getLogger(__name__)

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

def training_job(config: Dict, model: torch.nn.Module, train_iter: DataLoader, valid_iter: DataLoader, test_iter: DataLoader, output_path: str, weights: List = None):
    """
    Main training loop
    """
    logger.info(f"Text Classification Training Pipeline")
    logger.info("Configuration")
    logger.info(json.dumps(config, indent=4))

    model = model.to(device)

    steps_per_epoch = len(train_iter)

    loss_function = cross_entropy_loss(weights)

    lr = config['learning-rate-pretrain']
    n_epochs = config['num-epochs-pretrain']
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch*1, num_training_steps=steps_per_epoch*n_epochs)

    pretrain(
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        valid_period=len(train_iter),
        num_epochs=n_epochs
    )

    lr = config['learning-rate-train']
    n_epochs = config['num-epochs-train']
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch * 1, num_training_steps=steps_per_epoch * n_epochs)
    validation_period = len(train_iter) // 1
    logger.info(f"Validation period: every '{validation_period}' batches")
    train(
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        valid_period=validation_period,
        num_epochs=n_epochs,
        output_path=output_path
    )

    write_train_valid_loss(output_path)

    return output_path


def pipeline(datapath: any, modelname: str, output_dir: str, stratify_by='label', config: dict={}) -> Tuple:
    """
    Main function that orchestrates the training workflow
    """
    logger.info(f"Main pipeline")
    logger.info(f"Stratifying pipeline by '{stratify_by}'")

    if type(datapath) == str:
        df = pd.read_csv(datapath)
    else:
        df = datapath
    n_outputs = df['label'].nunique()

    tokenizer = create_tokenizer(modelname)

    model = make_model(modelname)(
        dropout_rate=config['dropout-ratio'],
        n_outputs=n_outputs
    )

    output_path = make_run_dir(output_dir)

    split_ratios = [0.8, 0.1, 0.1]
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer=tokenizer, filepath=datapath, split_ratios=split_ratios, stratify_by=stratify_by)

    logger.info(f"Saving datasets to '{output_path}'")

    torch.save(train_dataset, os.path.join(output_path, 'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(output_path, 'val_dataset.pt'))
    torch.save(test_dataset, os.path.join(output_path, 'test_dataset.pt'))

    train_iter, valid_iter, test_iter = make_iterators(train_dataset, val_dataset, test_dataset)

    weights = calculate_multiclass_weights(df['label'])

    training_job(
        config=config,
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        test_iter=test_iter,
        output_path=output_path,
        weights=weights
    )

    model = make_model(modelname)(
        n_outputs=n_outputs
    )

    model = load_model(model, output_path)
    y_true, y_pred = evaluate(model, test_iter, threshold=config['threshold'])

    classification_report(y_true, y_pred)
    write_confusion_matrix(y_true, y_pred, output_path)

    return model, y_true, y_pred, output_path, train_dataset, val_dataset, test_dataset

def predict(modelname, datapath, output_path, n_outputs=2, threshold=0.5):
    """
    Predict on test dataset
    """
    logger.info(f"Predicting on test dataset")
    model = make_model(modelname)(
        n_outputs=n_outputs
    )

    tokenizer = create_tokenizer(modelname)
    model = load_model(model, output_path)

    test_set = create_testset(tokenizer=tokenizer, filepath=datapath)
    test_iter = make_test_iterator(test_set)
    batch = next(iter(test_iter))
    batch = {k: v.to(device) for k, v in batch.items()}

    target = batch.pop('target')
    yhat = model(**batch)  # Give dummy batch to forward()
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

    y_true, y_pred = evaluate(model, test_iter, threshold=threshold)
    classification_report(y_true, y_pred)
    write_confusion_matrix(y_true, y_pred, output_path)

if __name__ == '__main__':

    from source.lm_classifier.arguments import args
    from source.lm_classifier.config import config

    datapath = args.data_path
    output_dir = args.output_dir
    modelname = args.model
    just_test = args.just_test
    print(just_test)

    if just_test:
        logger.info(f"Running in test mode")
        predict(modelname, datapath, output_dir)
    else:
        logger.info(f"Running in train mode")
        for arg, value in sorted(vars(args).items()):
            logging.info(f"Argument {arg}: '{value}'")

        model, y_true, y_pred, output_path, train_dataset, val_dataset, test_dataset = pipeline(
            datapath=datapath,
            modelname=modelname,
            output_dir=output_dir,
            config=config
        )

        score = f1_score(y_true, y_pred, average='weighted')
        logger.info(f"weighted f1-score '{score}'")


