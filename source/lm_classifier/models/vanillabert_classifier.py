import torch

from transformers import AutoTokenizer
from transformers import AutoModel


class VanillaBertClassifier(torch.nn.Module):
    """
    Vanilla BERT model for text classification tasks (e.g., sentiment analysis, topic classification, etc.)
    """

    def __init__(self, dropout_rate=0.3, n_outputs=2):
        super(VanillaBertClassifier, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained("bert-base-uncased")
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, n_outputs)

    def forward(self, *args, **kwargs):
        _, x = self.pretrained_model(*args, **kwargs, return_dict=False)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)

        return x