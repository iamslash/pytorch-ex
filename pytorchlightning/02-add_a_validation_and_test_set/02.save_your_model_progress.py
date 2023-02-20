import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl

##################################################
# Checkpoints include these
#
# * 16-bit scaling factor (if using 16-bit precision training)
# * Current epoch
# * Global step
# * LightningModule’s state_dict
# * State of all optimizers
# * State of all learning rate schedulers
# * State of all callbacks (for stateful callbacks)
# * State of datamodule (for stateful datamodules)
# * The hyperparameters used for that model if passed in as hparams (Argparse.Namespace)
# * The hyperparameters used for that datamodule if passed in as hparams (Argparse.Namespace)
# * State of Loops (if using Fault-Tolerant training)

##################################################
# Save a checkpoint

# Define the PyTorch nn.Modules
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

# Define a LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Define the training datatset
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)

# Train the model
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
# trainer = pl.Trainer(default_root_dir="/tmp/chkpts/")
trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

##################################################
# LightningModule from checkpoint
model = LitAutoEncoder.load_from_checkpoint("/tmp/chkpts/checkpoint.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(dataset.data)

# Save hyperparameters
class MyLightningModule(pl.LightningModule):
    def __init__(self, learning_rate, another_parameter, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])
# {"learning_rate": the_value, "another_parameter": the_other_value}

# The LightningModule also has access to the Hyperparameters
model = MyLightningModule.load_from_checkpoint("/tmp/chkpts/checkpoint.ckpt")
print(model.learning_rate)

# Initialize with other parameters

# if you train and save the model like this it will use these values when loading
# the weights. But you can overwrite this
CKPT_PATH = "/tmp/chkpts/checkpoint.ckpt"
LitModel(in_dim=32, out_dim=10)

# uses in_dim=32, out_dim=10
model = LitModel.load_from_checkpoint(CKPT_PATH)

# uses in_dim=128, out_dim=10
model = LitModel.load_from_checkpoint(CKPT_PATH, in_dim=128, out_dim=10)

##################################################
# nn.Module from checkpoint

checkpoint = torch.load(CKPT_PATH)
print(checkpoint.keys())

checkpoint = torch.load(CKPT_PATH)
encoder_weights = checkpoint["encoder"]
decoder_weights = checkpoint["decoder"]

##################################################
# Disable checkpointing

trainer = pl.Trainer(enable_checkpointing=False)

##################################################
# Resume training state

model = MyLightningModule()
trainer = pl.Trainer()

# automatically restores model, epoch, step, LR schedulers, etc...
trainer.fit(model, ckpt_path=CKPT_PATH)
