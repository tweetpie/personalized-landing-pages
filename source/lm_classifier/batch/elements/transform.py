import logging
import networkx as nx

from itertools import repeat

from typing import Dict
from typing import List

from source.lm_classifier.batch.ops.graph.nodes import get_articles, split_blocks
from source.lm_classifier.batch.ops.graph.remove import remove_schedules
from source.lm_classifier.batch.ops.graph.remove import remove_irrelevant_nodes
from source.lm_classifier.constants import MODEL_NAME
from source.lm_classifier.io.model import load_model
from source.lm_classifier.models import make_model
from source.lm_classifier.ops.prediction import predict_texts
from source.lm_classifier.tokenizer import create_tokenizer


logger = logging.getLogger(__name__)


def process_single(file: str, doc: Dict, g: nx.DiGraph):
    """
    Processes a single document
    Returns a Document-level Burden Object

    Steps:
        * Segregates by article
        * Segregates each article by blocks
        * Runs inference process at the block level
        * Aggregates results at the block level
        * Creates an object representation
    """
    g = remove_schedules(g)
    g = remove_irrelevant_nodes(g)
    articles = get_articles(g)

    blocks = [split_blocks(g, a) for a in articles]

    identifier = doc['metadata']['identifier']

    items = [
        list(zip(repeat(identifier), repeat(g.node[a]['id']), bs))
        for a, bs in zip(articles, blocks)
    ]
    items = sum(items, [])
    items = [(d, a) + bs for d, a, bs in items]

    return items


def run_inference(texts: List[str], model_dir: str, n_outputs: int):
    """
    Runs an inference job using the model in `model_dir`
    """
    logger.info(f"Running an inference job to infer '{len(texts)}' documents")

    modelname = MODEL_NAME.ROBERTA_POOLED

    tokenizer = create_tokenizer(modelname)

    model = make_model(modelname)(
        n_outputs=n_outputs
    )

    model = load_model(model, model_dir)
    model.eval()

    y_preds, y_probs = predict_texts(model, tokenizer, texts)

    return y_preds, y_probs
