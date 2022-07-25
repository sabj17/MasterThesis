import torch.nn as nn
from einops.layers.torch import Reduce


class BaseHead(nn.Module):
    def __init__(self, layer_func, in_dim, hidden_dim, out_dim, depth):
        super().__init__()

        layers = []
        for i in range(depth):
            dim = in_dim if i == 0 else hidden_dim

            layer = layer_func(dim, hidden_dim)
            layers.append(layer)

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class Head(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=1):
        super().__init__()

        def layer_func(in_dim, out_dim): return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU()
        )

        self.head = BaseHead(layer_func, in_dim, hidden_dim, out_dim, depth)

    def forward(self, x):
        return self.head(x)


class ChannelWiseHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=1):
        super().__init__()
        from einops import rearrange

        def layer_func(in_dim, out_dim): return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

        self.head = BaseHead(layer_func, 32, 128, 22, depth)

    def forward(self, x):
        x = rearrange(x, 'b s (c p) ->  b s p c', p=5)
        self.head(x)
        x = rearrange(x, 'b s p c ->  b s c p', p=5)
        return x


class HeadWithLayerNorm(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=1):
        super().__init__()

        def layer_func(in_dim, out_dim): return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

        self.head = BaseHead(layer_func, in_dim, hidden_dim, out_dim, depth)

    def forward(self, x):
        return self.head(x)


class TransformerClassificationHead(nn.Module):
    def __init__(self, embed_size, n_classes, hidden_factor=0):
        super().__init__()

        self.head = nn.Sequential(
            Reduce('b c e -> b e', 'mean'),
            nn.Linear(embed_size, 4),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.head(x)


class FlattenHead(nn.Module):
    def __init__(self, input_size, embed_size, n_classes):
        super().__init__()

        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(input_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Linear(embed_size, n_classes),
        )

    def forward(self, x):
        print(x.shape)
        return self.head(x)


def get_head(head_info, embed_size, data_exp):
    factories = {
        "head1": Head,
        "head_layernorm": HeadWithLayerNorm,
        "channelhead": ChannelWiseHead
    }

    head_name = head_info['type']
    params = head_info['parameters']

    if 'in_dim' in params:
        in_dim = params['in_dim']
    else:
        in_dim = embed_size

    hidden_dim = int(in_dim*params['hidden_factor'])

    if 'out_dim' in params:
        out_dim = params['out_dim']
    else:
        out_dim = embed_size

    if head_name in factories:
        return factories[head_name](in_dim, hidden_dim, out_dim, params['depth'])
    print(f"Unknown head option: {head_name}.")
