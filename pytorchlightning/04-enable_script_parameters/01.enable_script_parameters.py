from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--layer_1_dim", type=int, default=128)
args = parser.parse_args()

# python *.py --layer_1_dim 64
