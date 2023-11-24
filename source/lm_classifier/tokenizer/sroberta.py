from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers import RobertaTokenizer, RobertaTokenizerFast


def make_sentence_roberta_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/roberta-base-nli-mean-tokens')
    return tokenizer



