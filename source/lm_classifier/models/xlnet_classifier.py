import torch

from transformers import XLNetModel
from transformers.activations import get_activation
from transformers.modeling_utils import SequenceSummary


class XLNETClassifier(torch.nn.Module):

    """
    XLNet is a method of pretraining language representations developed by CMU and Google researchers in mid-2019.
    XLNet was created to address what the authors saw as the shortcomings of the autoencoding method of
    pretraining used by BERT and other popular language models. We won’t get into the details of XLNet
    in this post, but the authors favored a custom autoregressive method.
    This pretraining method resulted in models that outperformed BERT on a range of
    NLP tasks and resulted in a new state of the art model.

    Input:
    BERT
        [CLS] + Sentence_A + [SEP] + Sentence_B + [SEP]
    XLNET
        Sentence_A + [SEP] + Sentence_B + [SEP] + [CLS]
    """

    def __init__(self, dropout_rate=0.3, n_outputs=2):
        super(XLNETClassifier, self).__init__()
        self.pretrained_model = XLNetModel.from_pretrained("xlnet-base-cased")
        self.sequence_summary = SequenceSummary(self.pretrained_model.config)
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, n_outputs)

    def forward(self, *args, **kwargs):
        x = self.pretrained_model(*args, **kwargs)[0]
        x = self.sequence_summary(x)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)

        return x