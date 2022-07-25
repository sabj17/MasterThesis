import torch
import torch.nn as nn
import numpy as np


class RandomTokenMasking(nn.Module):
    def __init__(self, embed_size, mask_pct=0.6):
        super().__init__()
        self.mask_pct = mask_pct
        self.mask_token = nn.Parameter(torch.randn(embed_size))

    def forward(self, x):
        batch_size, n_tokens = x.size(0), x.size(1)
        device = x.get_device() if x.get_device() != -1 else 'cpu'

        mask = torch.stack(
            [create_random_mask(n_tokens, self.mask_pct, device=device)
             for _ in range(batch_size)]
        )

        x_masked = x.detach().clone()
        x_masked[mask] = self.mask_token

        return x_masked, mask


class RandomZeroMasking(nn.Module):
    def __init__(self, embed_size, mask_pct=0.6):
        super().__init__()
        self.mask_pct = mask_pct

    def forward(self, x):
        device = x.get_device() if x.get_device() != -1 else 'cpu'

        mask_props = torch.ones(x.size(0), x.size(1), device=device)
        mask_props *= self.mask_prop
        mask = torch.bernoulli(mask_props).bool()

        x_masked = x.detach().clone()
        x_masked[mask] = torch.zeros_like(x_masked[mask]).bool()

        return x_masked, mask


class CutoutTokenMasking(nn.Module):
    def __init__(self, embed_size, mask_pct=0.6):
        super().__init__()
        self.mask_pct = mask_pct
        self.mask_token = nn.Parameter(torch.randn(embed_size))

    def forward(self, x):
        batch_size, n_tokens = x.size(0), x.size(1)
        device = x.get_device() if x.get_device() != -1 else 'cpu'

        mask = torch.stack(
            [create_cutout_mask(n_tokens, self.mask_pct, device=device)
             for _ in range(batch_size)]
        )

        x_masked = x.detach().clone()
        x_masked[mask] = self.mask_token

        return x_masked, mask


class MixedMasking(nn.Module):
    def __init__(self, embed_size, mask_pct=0.6, ratio=0.5):
        super().__init__()
        self.mask_pct = mask_pct
        self.mask_token = nn.Parameter(torch.randn(embed_size))
        self.ratio = ratio

    def forward(self, x):
        batch_size, n_tokens = x.size(0), x.size(1)
        device = x.get_device() if x.get_device() != -1 else 'cpu'

        mask = []
        for _ in range(batch_size):

            if np.random.binomial(1, self.ratio) == 1:
                sample_mask = create_random_mask(
                    n_tokens, self.mask_pct, device=device)
            else:
                sample_mask = create_cutout_mask(
                    n_tokens, self.mask_pct, device=device)

            mask.append(sample_mask)

        mask = torch.stack(mask)

        x_masked = x.detach().clone()
        x_masked[mask] = self.mask_token

        return x_masked, mask


def create_random_mask(n_tokens, mask_pct, device='cpu'):
    mask_props = torch.ones(n_tokens, device=device)
    mask_props *= mask_pct
    mask = torch.bernoulli(mask_props).bool()

    return mask


def create_cutout_mask(n_tokens, mask_pct, device='cpu'):
    mask_len = int(mask_pct * n_tokens)

    mask = torch.zeros(n_tokens, device=device).bool()

    start_idx = np.random.randint(0, n_tokens - mask_len)
    mask[start_idx: start_idx+mask_len] = True

    return mask


def get_masker(mask_info, embed_size):
    factories = {
        "random": RandomTokenMasking,
        "cutout": CutoutTokenMasking,
        "mixed": MixedMasking,
    }

    mask_name = mask_info["type"]

    if mask_name in factories:
        return factories[mask_name](embed_size, mask_info["mask_pct"])
    print(f"Unknown framework option: {mask_name}.")
