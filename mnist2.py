# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
# ---

# %% Imports

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = './datasets'
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

# %% Lit Module with data

class LitMNIST(LightningModule):

    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        self.save_hyperparameters()
        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def evaluate_batch(self, batch):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        return x, y, logits, preds
    def validation_step(self, batch, batch_idx):
        x, y, logits, preds = self.evaluate_batch(batch)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', F.nll_loss(logits, y), prog_bar=True)
        self.log('val_acc', accuracy(preds, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        x, y, logits, preds = self.evaluate_batch(batch)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('test_loss', F.nll_loss(logits, y), prog_bar=True)
        self.log('test_acc', accuracy(preds, y), prog_bar=True)
        return x, y, logits, preds

    def test_epoch_end(self, outputs):
        # outputs = [(x1, y1, preds1), (x2, y2, preds2), ...]
        x, y, logits, preds = (torch.cat(l, dim=0) for l in zip(*outputs))
        loss_test = F.nll_loss(logits, y)
        acc_test = accuracy(preds, y)

        self.logger.log_hyperparams(
            {**self.hparams},
            {'test_acc': acc_test, 'test_loss': loss_test},
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


# %% Training

model = LitMNIST()
trainer = Trainer(
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False),
    max_epochs=1,
    progress_bar_refresh_rate=20,
)
trainer.fit(model)

# %% Testing

trainer.test()
