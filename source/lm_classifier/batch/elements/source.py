import re
import logging
import pathlib

import networkx as nx

from typing import Dict
from typing import Generator
from typing import Tuple

from source.lm_classifier.batch.ops.doc import get_category
from source.lm_classifier.batch.ops.io.files import read_json
from source.lm_classifier.batch.ops.io.wm import from_data

logger = logging.getLogger(__name__)


Item = Tuple[str, Dict, nx.DiGraph]


def traverse_documents(input_dir: str, n: int = None) -> Generator[Item, None, None]:
    """
    Returns a stream for all of the documents in `input_dir`
    """
    logger.info(f"Reading documents in '{input_dir}'")

    files = list(pathlib.Path(input_dir).glob('**/*.json'))
    files = sorted(files, reverse=True)
    logger.info(f"Found '{len(files)}' files")

    # files = [
    #     '/Users/raulferreira/waymark/data/prepared/enacted-epublished-xml/1999/uksi-1999-3242-enacted-data.json'
    # ]

    if n is None:
        n = len(files) + 1

    pattern = re.compile(r'/\d+/(?P<cat>.+)-(?P<year>\d+)-(?P<number>\d+)')

    for i, file in enumerate(files[:n]):

        if (i % 10000) == 0:
            logger.info(f"Progress: '{i} / {len(files)}'")

        file = str(file)
        try:
            doc = read_json(file)
        except:
            logger.error(f"Error while parsing file '{file}'")
            continue

        match = pattern.search(file)
        cat = match.group('cat')
        year = match.group('year')
        number = match.group('number')

        category = get_category(doc)

        if category not in ['UnitedKingdomPublicGeneralAct', 'UnitedKingdomStatutoryInstrument']:
            continue

        g = from_data(doc)

        yield file, doc, g


