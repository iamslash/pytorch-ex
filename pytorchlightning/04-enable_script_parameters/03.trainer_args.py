from argparse import ArgumentParser

from pytorch_lightning import Trainer

parser = ArgumentParser()
parser = Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

trainer = Trainer.from_argparse_args(hparams)

# # or if you need to pass in callbacks
# trainer = Trainer.from_argparse_args(hparams, enable_checkpointing=..., callbacks=[...])
