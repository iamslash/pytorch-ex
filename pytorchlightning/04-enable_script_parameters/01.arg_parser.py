from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--layer_1_dim", type=int, default=128)
args = parser.parse_args()
print(args)

# python 01.arg_parser.py --layer_1_dim=256
# Namespace(layer_1_dim=256)


