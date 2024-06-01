import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from baselines.utils import aleatoric_loss, compute_uncertainty


class DEModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, n_ensembles=5, dropout=0.3):
        super(DEModel, self).__init__()
        self.models = torch.nn.ModuleList(
            [model(num_classes=num_classes, dropout=dropout, monte_carlo=False, aleatoric=True) for _ in range(n_ensembles)])
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None

    def forward(self, inputs):
        return [model(inputs) for model in self.models]

    def training_step(self, batch, batch_idx):
        image, audio, text, target = batch
        outputs = self((image, audio, text))
        loss = 0
        for output in outputs:
            logits, sigma = output
            loss += aleatoric_loss(logits, target, sigma, 100)
        self.log('train_loss', loss)
        logits = torch.stack([output[0] for output in outputs], dim=1).mean(dim=1)
        acc = self.train_acc(logits, target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def val_test_shared_step(self, batch):
        image, audio, text, target = batch
        outputs = self((image, audio, text))
        loss = 0
        for output in outputs:
            logits, sigma = output
            loss += aleatoric_loss(logits, target, sigma, 100)
        outputs = torch.stack([output[0] for output in outputs], dim=1)
        outputs_mu = outputs.mean(dim=1)
        log_sigma = torch.stack([output[1] for output in outputs], dim=1)
        # sigma = torch.exp(log_sigma)
        # log_sigma = torch.log(sigma.mean(dim=1))
        log_sigma = log_sigma.mean(dim=1)

        entropy_ale, entropy_ep = compute_uncertainty(outputs_mu, log_sigma, torch.log(outputs.std(dim=1)))
        return loss, outputs_mu, target, entropy_ale, entropy_ep

    def test_step(self, batch, batch_idx):
        loss, output, target, entropy_ale, entropy_ep = self.val_test_shared_step(batch)
        self.log('test_loss', loss)
        self.test_acc(output, target)
        return loss, entropy_ale, entropy_ep

    def validation_step(self, batch, batch_idx):
        loss, output, target, entropy_ale, entropy_ep = self.val_test_shared_step(batch)
        self.val_acc(output, target)
        return loss, entropy_ale, entropy_ep

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_loss', torch.stack([x[0] for x in outputs], dim=0).mean(), prog_bar=True)
        self.log('val_entropy_ale', torch.cat([x[1] for x in outputs], dim=0).mean(), prog_bar=True)
        self.log('val_entropy_epi', torch.cat([x[2] for x in outputs], dim=0).mean(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_ale', torch.cat([x[1] for x in outputs], dim=0).mean(), prog_bar=True)
        self.log('test_entropy_epi', torch.cat([x[2] for x in outputs], dim=0).mean(), prog_bar=True)
        self.aleatoric_uncertainties = torch.cat([x[1] for x in outputs], dim=0).detach().cpu().numpy()
        self.epistemic_uncertainties = torch.cat([x[2] for x in outputs], dim=0).detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
