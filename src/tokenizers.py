import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat


class EEGtoPatchTokens(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        assert x.dim() == 3  # (batch_size, channels, seq_len)

        spill = x.size(-1) % self.patch_size

        if spill != 0:
            pad_size = self.patch_size - spill

            device = x.get_device() if x.get_device() != -1 else 'cpu'
            pad = torch.zeros(x.size(0), x.size(1), pad_size, device=device)

            x = torch.concat([x, pad], dim=-1)

        x = rearrange(x, 'b c (s p) ->  b s (c p)', p=self.patch_size)
        return x


class EEGtoToken(nn.Module):
    def __init__(self, n_channels, seq_len, slice_size):
        super().__init__()
        assert seq_len % slice_size == 0

        self.slice_size = slice_size
        self.n_channels = n_channels
        self.seq_len = seq_len

    def forward(self, x):
        assert x.dim() == 3  # (batch_size, channels, seq_len)
        # (batch_size, channels, n_slices, slice_size)
        x = x.unfold(-1, self.slice_size, self.slice_size)
        # (batch_size, channels * n_slices, slice_size)
        x = x.flatten(start_dim=1, end_dim=2)
        return x

    @property
    def n_slices(self):
        return int(self.seq_len / self.slice_size) * self.n_channels
