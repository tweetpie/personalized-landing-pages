from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer


def make_legalbert_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    return tokenizer


