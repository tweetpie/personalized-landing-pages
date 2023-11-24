from transformers import PreTrainedTokenizer
from source.lm_classifier.constants import MODEL_NAME

from source.lm_classifier.tokenizer.electra import make_electra_tokenizer
from source.lm_classifier.tokenizer.legalbert import make_legalbert_tokenizer
from source.lm_classifier.tokenizer.roberta import make_roberta_tokenizer
from source.lm_classifier.tokenizer.sbert import make_sentence_bert_tokenizer
from source.lm_classifier.tokenizer.sroberta import make_sentence_roberta_tokenizer
from source.lm_classifier.tokenizer.xlnet import make_xlnet_tokenizer
from source.lm_classifier.tokenizer.vanillabert import make_vanillabert_tokenizer


def create_tokenizer(modelname: str) -> PreTrainedTokenizer:
    """
    Creates the correct tokenizer given the model name
    """

    if modelname in [MODEL_NAME.ROBERTA, MODEL_NAME.ROBERTA_POOLED]:
        return make_roberta_tokenizer()

    elif modelname == MODEL_NAME.ELECTRA:
        return make_electra_tokenizer()

    elif modelname == MODEL_NAME.LEGALBERT:
        return make_legalbert_tokenizer()

    elif modelname == MODEL_NAME.XLNET:
        return make_xlnet_tokenizer()

    elif modelname == MODEL_NAME.VANILLABERT:
        return make_vanillabert_tokenizer()

    elif modelname == MODEL_NAME.SBERT:
        return make_sentence_bert_tokenizer()

    elif modelname == MODEL_NAME.SROBERTA:
        return make_sentence_roberta_tokenizer()

    else:
        raise NotImplementedError(f"Unknown model '{modelname}'")
