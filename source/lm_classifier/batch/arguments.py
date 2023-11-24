import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--output-dir", required=True)
parser.add_argument("--input-dir", required=True)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--n-outputs", required=True, type=int)

args = parser.parse_args()