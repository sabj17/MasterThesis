import torch
import torch.nn.functional as F
import torch.nn as nn

class NT_Xent(nn.Module):
    def __init__(self, temperature=0.5):
        super(NT_Xent, self).__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        assert len(z1) == len(z2)
        N = 2 * len(z1)

        device = z1.get_device() if z1.get_device() != -1 else 'cpu'

        # Cosine similarity
        z1, z2 = F.normalize(z1), F.normalize(z2) # Normalized to compute cosine similiarty
        z = torch.cat((z1, z2), dim=0)
        sim = torch.exp(torch.mm(z, z.t()) / self.tau) #

        # Compute negative similarity by masking out diagonal and summing rows
        mask = ~torch.eye(N, device=device).bool()
        neg = sim.masked_select(mask).view(N, -1)
        neg = torch.sum(neg, dim=-1)

        # Compute positive similarity by dot product of normalized outputs
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / self.tau)
        pos = torch.cat([pos, pos])

        loss = -torch.log(pos / neg)
        loss = torch.mean(loss)
        
        return loss

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, z1, z2):
        loss = F.smooth_l1_loss(z1, z2, reduction="none", beta=self.beta)
        loss = loss.sum(dim=-1).mean()
        
        return loss

class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return 2 - 2 * (z1 * z2).sum(dim=-1).mean()


def get_criterion(loss_info):
    factories = {
        "simclr" : NT_Xent,
        "byol" : BYOLLoss,
        "data2vec": SmoothL1Loss,
        "supervised": nn.CrossEntropyLoss,
        "l2": nn.MSELoss,
        "l1": nn.L1Loss
    }
    
    if loss_name in factories:
        return factories[loss_name]()
    print(f"Unknown loss option: {loss_name}.")