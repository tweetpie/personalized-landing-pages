import argparse

from source.lm_classifier.constants import MODELS


parser = argparse.ArgumentParser(description='Text Classification')

parser.add_argument("--model", help='Which model to use', choices=MODELS, required=True)
parser.add_argument("--output-dir", help='Base directory to save outputs of the run', required=True)
parser.add_argument("--data-path", help='Location of the CSV with `text` and `label`', required=True)
parser.add_argument("--just-test", help='Just test the model')
args = parser.parse_args()