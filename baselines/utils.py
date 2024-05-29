import torch
import torch.nn.functional as F
from torch import nn


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
    # std = torch.exp(log_std)
    std = log_std
    mu_mc = logits.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples)
    # hard coded the known shape of the data
    noise = torch.randn(*logits.shape, num_samples, device=logits.device) * std.unsqueeze(-1)
    prd = mu_mc + noise

    targets = targets.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples).squeeze(0)
    mc_x = torch.nn.functional.cross_entropy(prd, targets, reduction='none')
    # mean across mc samples
    mc_x = mc_x.mean(-1)
    # mean across every thing else
    mc_x_mean = mc_x.mean()
    # assert is not inf or nan
    assert not torch.isfinite(mc_x_mean).sum() == 0, f"Loss is inf: {mc_x_mean}"
    return mc_x.mean()


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def get_dc_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum


class AvgTrustedLoss(nn.Module):
    def __init__(self, num_views: int, annealing_start=50, gamma=1):
        super(AvgTrustedLoss, self).__init__()
        self.num_views = num_views
        self.annealing_step = 0
        self.annealing_start = annealing_start
        self.gamma = gamma

    def forward(self, evidences, target, evidence_a, **kwargs):
        num_classes = evidences.shape[-1]
        target = F.one_hot(target, num_classes)
        alphas = evidences + 1
        loss_acc = edl_digamma_loss(alphas, target, self.annealing_step, num_classes, self.annealing_start,
                                    evidence_a.device)
        for v in range(len(evidences)):
            alpha = evidences[v] + 1
            loss_acc += edl_digamma_loss(alpha, target, self.annealing_step, num_classes, self.annealing_start,
                                         evidence_a.device)
        loss_acc = loss_acc / (len(evidences) + 1)
        loss = loss_acc + self.gamma * get_dc_loss(evidences, evidence_a.device)
        return loss


def sampling_softmax(logits, log_sigma, num_samples=100):
    std = torch.exp(log_sigma)
    mu_mc = logits.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples)
    # hard coded the known shape of the data
    noise = torch.randn(*logits.shape, num_samples, device=logits.device) * std.unsqueeze(-1)
    prd = mu_mc + noise
    return torch.softmax(prd, dim=0).mean(-1)


def compute_uncertainty(outputs, log_sigmas_ale, log_sigmas_ep, num_samples=100):
    p_ale = sampling_softmax(outputs, log_sigmas_ale, num_samples)
    entropy_ale = -torch.sum(p_ale * torch.log(p_ale + 1e-6), dim=-1)
    p_ep = sampling_softmax(outputs, log_sigmas_ep, num_samples)
    entropy_ep = -torch.sum(p_ep * torch.log(p_ep + 1e-6), dim=-1)
    return entropy_ale.mean(-1), entropy_ep.mean(-1)
