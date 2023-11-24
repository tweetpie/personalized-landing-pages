import os

import numpy as np

import torch
import logging
import warnings

# __path__ = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

file_format ='%(asctime)s, %(name)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
console_format ='%(asctime)s, %(name)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'

logging.basicConfig(
    # format=file_format,
    # datefmt=console_format,
    level=logging.INFO,
    filename='training-job.log',
    filemode='w'
)

formatter = logging.Formatter(console_format)
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

torch.manual_seed(4)
np.random.seed(2)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')

logger = logging.getLogger(__name__)

logger.info(f"Device detected '{device}'")