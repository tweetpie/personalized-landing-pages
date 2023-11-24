from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers import RobertaTokenizer, RobertaTokenizerFast


def make_sentence_bert_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    return tokenizer


