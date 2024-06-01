import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from baselines.utils import aleatoric_loss, compute_uncertainty


class MCDModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, mc_samples=100, dropout=0.3):
        super(MCDModel, self).__init__()
        self.mc_samples = mc_samples
        self.model = model(num_classes=num_classes, dropout=dropout, monte_carlo=True if mc_samples > 1 else False)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None

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

    def val_test_shared_step(self, batch):
        image, audio, text, target = batch
        outputs = []
        sigmas = []
        for _ in range(self.mc_samples):
            output, sigma = self((image, audio, text))
            outputs.append(output)
            sigmas.append(sigma)
        output = torch.stack(outputs, dim=1)
        sigma = torch.stack(sigmas, dim=1)
        sigma = sigma.mean(dim=1)
        output_mu = output.mean(dim=1)
        loss = aleatoric_loss(output_mu, target, sigma, 100)

        entropy_ale, entropy_ep = compute_uncertainty(output_mu, sigma, torch.log(output.std(dim=1)))
        return loss, output_mu, target, entropy_ale, entropy_ep

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
