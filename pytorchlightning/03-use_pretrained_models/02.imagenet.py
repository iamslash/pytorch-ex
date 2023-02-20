import torch
import torchvision.models as models
import pytorch_lightning as pl

CKP_PATH = "/tmp/chkpts/checkpoint.ckpt"


class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = pl.nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = pl.nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)


def some_images_from_cifar10():
    raise NotImplementedError()


# Finetune
model = ImagenetTransferLearning()
trainer = pl.Trainer()
trainer.fit(model)

# predict your data of interest
model = ImagenetTransferLearning.load_from_checkpoint(CKP_PATH)
model.freeze()

x = some_images_from_cifar10()
predictions = model(x)
