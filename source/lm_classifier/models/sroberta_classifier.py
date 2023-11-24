import torch
from sentence_transformers import SentenceTransformer

from transformers import RobertaModel, AutoModel


class SRobertaClassifier(torch.nn.Module):

    """
    Sentence-RoBERTa model for text classification tasks (e.g., sentiment analysis, topic classification, etc.)

    """

    def __init__(self, dropout_rate=0.3, n_outputs=2):
        super(SRobertaClassifier, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained('sentence-transformers/roberta-base-nli-mean-tokens')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, n_outputs)

    def forward(self, *args, **kwargs):
        hidden_states, _ = self.pretrained_model(*args, **kwargs, return_dict=False)
        # here I use only representation of <s> token, but we can easily use more tokens,
        # custom pooling, etc
        # x = self.d1(hidden_states[:, 0, :])
        x = self.d1(hidden_states.mean(dim=1))
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)
        return x
