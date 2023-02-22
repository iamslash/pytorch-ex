from argparse import ArgumentParser

from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torchvision.models.vision_transformer import Encoder


class LitMNIST(LightningModule):
    def __init__(self, layer_1_dim, **kwargs):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, layer_1_dim)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitMNIST")
        parser.add_argument("--layer_1_dim", type=int, default=128)
        return parent_parser


class GoodGAN(LightningModule):
    def __init__(self, encoder_layers, **kwargs):
        super().__init__()
        self.encoder = Encoder(layers=encoder_layers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GoodGAN")
        parser.add_argument("--encoder_layers", type=int, default=12)
        return parent_parser


def main(args):
    dict_args = vars(args)

    # pick model
    if args.model_name == "gan":
        model = GoodGAN(**dict_args)
    elif args.model_name == "mnist":
        model = LitMNIST(**dict_args)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument("--model_name", type=str, default="gan", help="gan or mnist")

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == "gan":
        parser = GoodGAN.add_model_specific_args(parser)
    elif temp_args.model_name == "mnist":
        parser = LitMNIST.add_model_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)

# $ python main.py --model_name gan --encoder_layers 24
# $ python main.py --model_name mnist --layer_1_dim 128
