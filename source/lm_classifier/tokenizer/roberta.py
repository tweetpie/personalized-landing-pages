from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer, RobertaTokenizerFast


def make_roberta_tokenizer() -> PreTrainedTokenizer:
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    return tokenizer



