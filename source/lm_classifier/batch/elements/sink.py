import os
import re
import logging
import pathlib

import networkx as nx

from typing import Dict
from typing import Generator
from typing import Tuple

from source.lm_classifier.batch.ops.doc import get_category
from source.lm_classifier.batch.ops.io.files import read_json, write_json
from source.lm_classifier.batch.ops.io.wm import from_data

logger = logging.getLogger(__name__)


Item = Tuple[str, Dict, nx.DiGraph]


def write_object(output_dir: str, file: str, identifier: str, data_obj: Dict) -> None:
    """
    Writes a single Document level Themes Object
    """
    data = {
        'identifier': identifier,
        'data': data_obj
    }

    filename = os.path.basename(file)
    year = os.path.basename(os.path.dirname(file))
    base_dir = os.path.join(output_dir, year)

    if not os.path.exists(base_dir):
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    output_file = os.path.join(base_dir, filename)

    write_json(data, output_file)

