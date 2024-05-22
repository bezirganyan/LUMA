import torch


class MCDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return torch.nn.functional.dropout(x, p=self.p, training=True)