from pytorch_lightning import LightningModule, Trainer


class LitModel(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--encoder_layers", type=int, default=12)
        parser.add_argument("--data_path", type=str, default="/some/path")
        return parent_parser


# ----------------
# trainer_main.py
# ----------------
from argparse import ArgumentParser

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument("--conda_env", type=str, default="some_name")
parser.add_argument("--notification_email", type=str, default="will@email.com")

# add model specific args
parser = LitModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

# python trainer_main.py --accelerator 'gpu' --devices 2 --num_nodes 2 \
#   --conda_env 'my_env' --encoder_layers 12


