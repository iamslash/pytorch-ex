import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class LitModel(nn.LightningModule):
    def validation_step(self, batch, batch_idx):
        pass


model = LitModel()
trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model)

early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
trainer = pl.Trainer(callbacks=[early_stop_callback])


# In case you need early stopping in a different part of training,
# subclass EarlyStopping and change where it is called:
class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)
