# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: region_name,title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
# ---

# %%

# %% [md]
"""
Mostly taken from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/cifar10-baseline.html
"""

# %% Imports
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

# %% Global context
seed_everything(7)

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)


# %% Data
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

cifar10_dm = CIFAR10DataModule(
    data_dir='./datasets',
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    num_samples=200,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

# %% Model
def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

# %% Lightning

class LitModel(LightningModule):

    def __init__(
            self,
            lr=0.05,
            batch_size=BATCH_SIZE,
        ):
        super().__init__()

        self.save_hyperparameters() # The arguments are saved as hparams
        self.model = create_model()

    def forward(self, x): # Same as for a pytorch Module
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss) # Logs metrics in tensorboard
        return loss # Lightning takes care of opt.zero_grad(), loss.backward(), opt.step() 

    def evaluate(self, batch, stage=None): # NOT a lightning method
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self): # Configure opt and scheduler
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.hparams.batch_size
        scheduler_dict = {
            'scheduler': OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict} # lightning will call opt.step() and scheduler_dict['scheduler'].step()


# %% Instantiate model

model = LitModel(lr=0.05)
model.datamodule = cifar10_dm

# %% Instantiate trainer

trainer = Trainer(
    max_epochs=30,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False),
    callbacks=[ModelCheckpoint(filename='best.ckpt')],
)

# %% Train
trainer.fit(model, cifar10_dm)

# %% Test
trainer.test(model, datamodule=cifar10_dm)
