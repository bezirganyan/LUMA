import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from baselines.utils import MCDropout


class MCDModel(pl.LightningModule):
    def __init__(self, model, num_classes=42):
        super(MCDModel, self).__init__()
        self.model = model
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        image, audio, text, target = batch
        output = self((image, audio, text))
        loss = torch.nn.functional.cross_entropy(output, target)
        self.log('train_loss', loss)
        acc = self.train_acc(output, target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, audio, text, target = batch
        output = self((image, audio, text))
        loss = torch.nn.functional.cross_entropy(output, target)
        self.log('val_loss', loss)
        self.val_acc(output, target)
        return loss

    def test_step(self, batch, batch_idx):
        image, audio, text, target = batch
        output = self((image, audio, text))
        loss = torch.nn.functional.cross_entropy(output, target)
        self.log('test_loss', loss)
        self.test_acc(output, target)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
