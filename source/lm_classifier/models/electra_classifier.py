import torch

from transformers import ElectraModel, ElectraConfig
from transformers.activations import get_activation


class ElectraClassifier(torch.nn.Module):

    """
    The first token of every sequence is always a special classification token ([CLS]).
    The final hidden state corresponding to this token is used as the
    aggregate sequence representation for classification tasks.
    Sentence pairs are poacked together into a single sequence.

    The first token of every sequence is always a special classification token ([CLS]).
    The final hidden state corresponding to this token is used as the
    aggregate sequence representation for classification tasks
    """

    def __init__(self, dropout_rate=0.3, n_outputs=2):
        super(ElectraClassifier, self).__init__()

        self.pretrained_model = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(256, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, n_outputs)

    def forward(self, *args, **kwargs):
        x = self.pretrained_model(*args, **kwargs)[0][:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = get_activation("gelu")(x)
        x = self.d2(x)
        x = self.l2(x)

        return x