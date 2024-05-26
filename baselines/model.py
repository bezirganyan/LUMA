import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from baselines.utils import aleatoric_loss


class MCDModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, mc_samples=100):
        super(MCDModel, self).__init__()
        self.mc_samples = mc_samples
        self.model = model
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        image, audio, text, target = batch
        output, sigma = self((image, audio, text))
        loss = aleatoric_loss(output, target, sigma, 100)
        self.log('train_loss', loss)
        acc = self.train_acc(output, target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        entropy, loss, output, sigma, target = self.val_test_shared_step(batch)
        self.log('val_loss', loss)
        self.val_acc(output, target)
        self.log('val_std', sigma.mean(-1).mean())
        self.log('val_entropy', entropy)
        return loss, sigma.mean(-1).mean(), entropy

    def val_test_shared_step(self, batch):
        image, audio, text, target = batch
        outputs = []
        sigmas = []
        for _ in range(self.mc_samples):
            output, sigma = self((image, audio, text))
            outputs.append(output)
            sigmas.append(sigma)
        output = torch.stack(outputs)
        log_sigma = torch.stack(sigmas)
        sigma = torch.exp(log_sigma)
        log_sigma = torch.log(sigma.mean(dim=0))
        output = output.mean(dim=0)
        loss = aleatoric_loss(output, target, log_sigma, 100)
        sigma = torch.exp(log_sigma)
        probs = torch.softmax(output, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
        return entropy, loss, output, sigma, target

    def test_step(self, batch, batch_idx):
        entropy, loss, output, sigma, target = self.val_test_shared_step(batch)
        self.log('test_loss', loss)
        self.test_acc(output, target)
        self.log('test_std', sigma.mean(dim=-1).mean())
        self.log('test_entropy', entropy)
        return loss, sigma.mean(dim=-1).mean(), entropy

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_entropy', torch.stack([x[2] for x in outputs]).mean(), prog_bar=True)
        self.log('val_sigma', torch.stack([x[1] for x in outputs]).mean(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy', torch.stack([x[2] for x in outputs]).mean())
        self.log('test_simga', torch.stack([x[1] for x in outputs]).mean())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
