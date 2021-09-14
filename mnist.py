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
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# !pip install "pytorch-lightning==1.4.5" "torchmetrics>=0.3" "tensorboard==2.6" "torch==1.9" "torchvision==0.10"

# %% Imports

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sklearn.metrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = './datasets'

# %% Lit Module with data

class LitMNIST(LightningModule):

    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4, batch_size=64):
        super().__init__()
        self.example_input_array = torch.randn(1, 28, 28)
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.batch_size = batch_size
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

    def evaluate_batch(self, batch): # Not a lightning method
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return x, y, logits, preds

    def training_step(self, batch, batch_idx):
        x, y, logits, preds = self.evaluate_batch(batch)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, logits, preds = self.evaluate_batch(batch)
        loss = F.nll_loss(logits, y)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def on_test_epoch_start(self):
        # TODO 0: log test metrics in hparams tab
        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#logging-hyperparameters
        self.logger.log_hyperparams(self.hparams, {'test_loss':0, 'test_acc': 0})


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        x, y, logits, preds = self.evaluate_batch(batch)
        loss = F.nll_loss(logits, y)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)
        return x, y, logits, preds

    def test_epoch_end(self, outputs):
        # Test epoch end doc: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#test-epoch-end


        x, y, logits, preds = (torch.cat(l, dim=0) for l in zip(*outputs))

        # TODO 2: Log confusion matrix
        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.loggers.TensorBoardLogger.html#pytorch_lightning.loggers.TensorBoardLogger.experiment
        # https://www.tensorflow.org/tensorboard/image_summaries#building_an_image_classifier
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
        cm = sklearn.metrics.confusion_matrix(y, preds)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        self.logger.experiment.add_figure('Confusion Matrix', disp.figure_, global_step=self.current_epoch)

        # TODO 5: Visualize the images wrongly predicted with the highest confidence

        errs_idx = (y != preds)
        errs_img, errs_logits = x[errs_idx, ...], logits[errs_idx, ...]
        top3_errs_idx = errs_logits.max(1)[0].argsort(0)[-3:]
        self.logger.experiment.add_images('wrong prediction', errs_img[top3_errs_idx, ...])

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
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


# %% Training

model = LitMNIST()

# TODO 1: Run the training on a GPU
# https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#gpus

# TODO 3: Save the model weights with the best accuracy
# https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#automatic-saving
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
model_checkpoint_cb = ModelCheckpoint(monitor='val_acc')

# TODO 4: Log Model Graph in tensorboard
# https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.tensorboard.html#pytorch_lightning.loggers.tensorboard.TensorBoardLogger.params.log_graph

# TODO 5: Log the profile of a training step in tensorboard 
# https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html#pytorch-profiling
# https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#use-tensorboard-to-view-results-and-analyze-model-performance

trainer = Trainer(
    logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False, name='mnist', log_graph=True),
    callbacks=[model_checkpoint_cb],
    max_epochs=3,
    gpus=1,
    progress_bar_refresh_rate=10,
)
trainer.fit(model)

# %% Testing
trainer.test()

# %%
# %reload_ext tensorboard
# %tensorboard --logdir lightning_logs/
