import logging
import torch

from source.lm_classifier import device


logger = logging.getLogger(__name__)


def pretrain(model, optimizer, train_iter, valid_iter, loss_function, scheduler=None, valid_period=None, num_epochs=5):
    """
    Pretrains Linear Layers (does not train the pre-trained model)
    """
    logger.info(f"Starting 'pretrain' phase")

    for param in model.pretrained_model.parameters():
        param.requires_grad = False

    model.train()

    train_loss = 0.0
    valid_loss = 0.0
    global_step = 0

    for epoch in range(num_epochs):

        for batch in train_iter:

            batch = {k: v.to(device) for k, v in batch.items()}

            target = batch.pop('target')

            y_pred = model(**batch)

            loss = loss_function(y_pred, target)
            loss.backward()

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            train_loss += loss.item()
            global_step += 1

            if global_step % valid_period == 0:

                model.eval()

                with torch.no_grad():

                    for batch in valid_iter:

                        batch = {k: v.to(device) for k, v in batch.items()}

                        target = batch.pop('target')

                        y_pred = model(**batch)
                        loss = loss_function(y_pred, target)

                        valid_loss += loss.item()

                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)

                model.train()

                logger.info('Epoch [{}/{}], global step [{}/{}], PT Loss: {:.4f}, Val Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_iter),
                              train_loss, valid_loss))

                train_loss = 0.0
                valid_loss = 0.0

    for param in model.pretrained_model.parameters():
        param.requires_grad = True

    logger.info('Pre-training done!')
