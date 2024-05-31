import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from baselines.utils import AvgTrustedLoss


class DirichletModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, dropout=0.):
        super(DirichletModel, self).__init__()
        self.num_classes = num_classes
        self.model = model(num_classes=num_classes, monte_carlo=False, dropout=dropout, dirichlet=True)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.criterion = AvgTrustedLoss(num_views=3)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.log('train_loss', loss)
        acc = self.train_acc(output, target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def shared_step(self, batch):
        image, audio, text, target = batch
        output_a, output = self((image, audio, text))
        output = torch.stack(output)
        loss = self.criterion(output, target, output_a)
        return loss, output_a, target

    def validation_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.val_acc(output, target)
        alphas = output + 1
        probs = alphas / alphas.sum(dim=-1, keepdim=True)
        entropy = self.num_classes / alphas.sum(dim=-1)
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        aleatoric_uncertainty = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(alpha_0 + 1)), dim=-1)
        return loss, output, target, entropy, aleatoric_uncertainty

    def test_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.test_acc(output, target)
        alphas = output + 1
        probs = alphas / alphas.sum(dim=-1, keepdim=True)
        entropy = self.num_classes / alphas.sum(dim=-1)
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        aleatoric_uncertainty = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(alpha_0 + 1)), dim=-1)
        return loss, output, target, entropy, aleatoric_uncertainty

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.criterion.annealing_step += 1

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_loss', np.mean([x[0].detach().cpu().numpy() for x in outputs]), prog_bar=True)
        self.log('val_entropy', torch.cat([x[3] for x in outputs]).mean(), prog_bar=True)
        self.log('val_sigma', torch.cat([x[4] for x in outputs]).mean(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy_epi', torch.cat([x[3] for x in outputs]).mean())
        self.log('test_ale', torch.cat([x[4] for x in outputs]).mean())
        self.aleatoric_uncertainties = torch.cat([x[4] for x in outputs]).detach().cpu().numpy()
        self.epistemic_uncertainties = torch.cat([x[3] for x in outputs]).detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
