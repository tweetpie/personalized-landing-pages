import json

import logging
import pathlib

import networkx as nx

from typing import Dict
from typing import List

logger = logging.getLogger(__name__)


def read_json(file):
    with open(file) as f:
        return json.load(f)


def write_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)
    logger.debug(f"Written '{file}'")


def list_json_files(path):
    files = list(pathlib.Path(path).glob('**/*.json'))
    logger.debug(f"Found '{len(files)}' files")
    return files


