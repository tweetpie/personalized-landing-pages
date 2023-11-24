import torch

from transformers import AutoTokenizer
from transformers import AutoModel


class LegalBertClassifier(torch.nn.Module):
    """
    LEGAL-BERT is a family of BERT models for the legal domain,
    intended to assist legal NLP research, computational law, and
    legal technology applications. To pre-train the different variations of LEGAL-BERT,
    we collected 12 GB of diverse English legal text from several fields
    (e.g., legislation, court cases, contracts) scraped from publicly available resources.

    The pre-training corpora of LEGAL-BERT include:

        * 116,062 documents of EU legislation, publicly available from EURLEX (http://eur-lex.europa.eu), the repository of EU Law running under the EU Publication Office.
        * 61,826 documents of UK legislation, publicly available from the UK legislation portal (http://www.legislation.gov.uk).
        * 19,867 cases from European Court of Justice (ECJ), also available from EURLEX.
        * 12,554 cases from HUDOC, the repository of the European Court of Human Rights (ECHR) (http://hudoc.echr.coe.int/eng).
        * 164,141 cases from various courts across the USA, hosted in the Case Law Access Project portal (https://case.law).
        * 76,366 US contracts from EDGAR, the database of US Securities and Exchange Commission (SECOM) (https://www.sec.gov/edgar.shtml).
    """

    def __init__(self, dropout_rate=0.3, n_outputs=2):
        super(LegalBertClassifier, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, n_outputs)

        print('DROPOUT', dropout_rate)

    def forward(self, *args, **kwargs):
        _, x = self.pretrained_model(*args, **kwargs)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)

        return x