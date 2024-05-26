import torch


class MCDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return torch.nn.functional.dropout(x, p=self.p, training=True)


class AleatoricClassificationLoss(torch.nn.Module):
    def __init__(self, num_samples=100):
        super(AleatoricClassificationLoss, self).__init__()
        self.num_samples = num_samples

    def forward(self, logits, targets, log_std):
        return aleatoric_loss(logits, targets, log_std, num_samples=self.num_samples)


def aleatoric_loss(logits, targets, log_std, num_samples=100):
    std = torch.exp(log_std)
    mu_mc = logits.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples)
    # hard coded the known shape of the data
    noise = torch.randn(*logits.shape, num_samples, device=logits.device) * std.unsqueeze(-1)
    prd = mu_mc + noise

    targets = targets.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples).squeeze(0)
    mc_x = torch.nn.functional.cross_entropy(prd, targets, reduction='none')
    # mean across mc samples
    mc_x = mc_x.mean(-1)
    # mean across every thing else
    return mc_x.mean()
