from transformers import PreTrainedTokenizer
from transformers import XLNetTokenizer


def make_xlnet_tokenizer() -> PreTrainedTokenizer:
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    return tokenizer



