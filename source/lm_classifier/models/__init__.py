from source.lm_classifier.constants import MODEL_NAME

from source.lm_classifier.models.electra_classifier import ElectraClassifier
from source.lm_classifier.models.legalbert_classifier import LegalBertClassifier
from source.lm_classifier.models.roberta_pooled_classifier import RobertaPooledClassifier
from source.lm_classifier.models.roberta_classifier import RobertaClassifier
from source.lm_classifier.models.sbert_classifier import SBertClassifier
from source.lm_classifier.models.sroberta_classifier import SRobertaClassifier
from source.lm_classifier.models.xlnet_classifier import XLNETClassifier
from source.lm_classifier.models.vanillabert_classifier import VanillaBertClassifier


def make_model(modelname: str):
    """
    Returns the correct model based on the selection
    """
    if modelname == MODEL_NAME.ROBERTA:
        return RobertaClassifier

    elif modelname == MODEL_NAME.ROBERTA_POOLED:
        return RobertaPooledClassifier

    elif modelname == MODEL_NAME.ELECTRA:
        return ElectraClassifier

    elif modelname == MODEL_NAME.LEGALBERT:
        return LegalBertClassifier

    elif modelname == MODEL_NAME.XLNET:
        return XLNETClassifier

    elif modelname == MODEL_NAME.VANILLABERT:
        return VanillaBertClassifier

    elif modelname == MODEL_NAME.SBERT:
        return SBertClassifier

    elif modelname == MODEL_NAME.SROBERTA:
        return SRobertaClassifier


    else:
        raise NotImplementedError(f"Unknown model '{modelname}'")
