from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer


def make_vanillabert_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


