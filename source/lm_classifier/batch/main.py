import os
import logging
from itertools import islice
from itertools import repeat
from collections import defaultdict

from source.lm_classifier.batch.elements.sink import write_object
from source.lm_classifier.batch.elements.source import traverse_documents
from source.lm_classifier.batch.elements.transform import run_inference
from source.lm_classifier.batch.elements.transform import process_single
from source.lm_classifier.batch.ops.burden import document_burden_view

logger = logging.getLogger(__name__)


def pipeline(input_dir: str, output_dir: str, model_dir: str, n_outputs: int):

    logger.info("Running main pipeline")
    logger.info(f"Input dir '{input_dir}'")
    logger.info(f"Output dir '{output_dir}'")
    logger.info(f"Model dir '{model_dir}'")

    idx_to_labels = {
        0: 'Lower Order Detail',
        1: 'No Burden',
        2: 'Reporting',
        3: 'Standards'
    }
    bridges = ['Lower Order Detail']
    ignore = ['No Burden', 'Lower Order Detail']

    logger.info(f"Labels object '{idx_to_labels}'")
    logger.info(f"Labels to act as a brifge '{bridges}'")
    logger.info(f"Labels to ignore as burden initiators '{ignore}'")

    stream = traverse_documents(input_dir)
    stream = list((islice(stream, 10)))

    stream = (xs + (process_single(*xs),) for xs in stream)

    stream = (list(zip(repeat(file), xs)) for file, doc, g, xs in stream)

    data = sum(stream, [])
    data = [(file,) + xs for file, xs in data]

    texts = [os.linesep.join(x[-1]) + os.linesep for x in data]

    # y_pred, y_probs = run_inference(texts, model_dir, n_outputs)

    import pickle

    with open('y_pred.pkl', 'rb') as f:
        y_pred = pickle.load(f)

    with open('y_probs.pkl', 'rb') as f:
        y_probs = pickle.load(f)

    data = (
        (file, url, article_id, block_id, text, pred, probs)
        for (file, url, article_id, block_id, text), pred, probs in zip(data, y_pred, y_probs)
    )

    documents = defaultdict(lambda: dict())

    for file, url, article_id, block_id, text, pred, probs in data:
        documents[url]['file'] = file
        if 'articles' not in documents[url]:
            documents[url]['articles'] = []
        documents[url]['articles'].append((article_id, block_id, text, pred, probs))

    logger.info(f"'{len(documents)}' documents in scope to create Burden View")

    logger.info(f"Resolving burden instances at the document level")

    for i, (url, d) in enumerate(documents.items()):
        d['burden-instances'] = document_burden_view(d['articles'], idx_to_labels, bridges, ignore)

    n_written = 0

    for url, d in documents.items():
        if d['burden-instances']:
            n_written += 1
            write_object(output_dir, d['file'], url, d['burden-instances'])

    logger.info(f"Written '{n_written}' documents")
    logger.info(f"'{len(documents) - n_written}' documents without any burden instance")

    return stream


if __name__ == '__main__':

    from source.lm_classifier.batch.arguments import args

    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    n_outputs = args.n_outputs

    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: '{value}'")

    # input_dir = '/Users/raulferreira/waymark/data/prepared/enacted-epublished-xml'
    # output_dir = '/Users/raulferreira/waymark/data/nlp-outputs/regulatory-burden'
    # model_dir = ''
    # n_outputs = 4

    pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        n_outputs=n_outputs
    )
