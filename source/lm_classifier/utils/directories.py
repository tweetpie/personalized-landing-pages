import os
import logging

from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)


def make_run_dir(basedir: str) -> str:
    """
    Makes `run` directory in `basedir` to hold
    the output of a program run.

    Uses date and time for the directory name
    Returns the name of the newly created folder
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    directory = os.path.join(basedir, dt_string)
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created '{directory}'")
    return directory




