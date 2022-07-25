import torch
import torch.nn as nn
from einops import repeat, rearrange
from embedders import PositionalEncoding, PositionalEmbedding1D
from copy import deepcopy
############################## FNET ##############################


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class FNet(nn.Module):
    def __init__(self, embed_size, depth, hidden_dim, dropout=0., return_sides=False):
        super().__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList([])
        self.return_sides = return_sides
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_size, FNetBlock()),
                PreNorm(embed_size, MLP(embed_size, hidden_dim, dropout=dropout))
            ]))

        self.position_enc = PositionalEncoding(
            embed_size, max_len=1000)

    def forward(self, x):
        x = self.position_enc(x)

        side_outputs = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

            side_outputs.append(x)

        if self.return_sides:
            return x, torch.stack(side_outputs)

        return x


############################## Transformer ##############################
class BaseTransformer1(nn.Module):
    def __init__(self, embed_size, depth, hidden_dim, nhead=8, dropout=0.1, activation='gelu', return_sides=False):
        super().__init__()

        el = nn.TransformerEncoderLayer(embed_size, nhead=nhead, dim_feedforward=hidden_dim,
                                        dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.enc_modules = nn.ModuleList([deepcopy(el) for i in range(depth)])

        self.return_sides = return_sides
        self.embed_size = embed_size

        self.position_enc = PositionalEncoding(embed_size)

    def forward(self, x):
        x = self.position_enc(x)


        side_outputs = []
        
        for module in self.enc_modules:
            x = module(x)
            side_outputs.append(x)

        side_outputs = [out.detach().clone() for out in side_outputs]

        if self.return_sides:
            return x, torch.stack(side_outputs)

        return x


class BaseTransformer2(nn.Module):
    def __init__(self, embed_size, depth, hidden_dim, nhead=8, dropout=0.1, activation='gelu', return_sides=False):
        super().__init__()

        el = nn.TransformerEncoderLayer(embed_size, nhead=nhead, dim_feedforward=hidden_dim,
                                        dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.enc_modules = nn.ModuleList([deepcopy(el) for i in range(depth)])

        self.return_sides = return_sides
        self.embed_size = embed_size

        self.position_enc = PositionalEmbedding1D(embed_size)

    def forward(self, x):
        x = self.position_enc(x)


        side_outputs = []
        
        for module in self.enc_modules:
            x = module(x)
            side_outputs.append(x)

        side_outputs = [out.detach().clone() for out in side_outputs]

        if self.return_sides:
            return x, torch.stack(side_outputs)

        return x


class TransformerWithProjection(nn.Module):
    def __init__(self, embed_size, depth, hidden_dim, nhead=8, dropout=0.1, activation='gelu', return_sides=False):
        super().__init__()
        self.embed_size = embed_size
        self.return_sides = return_sides

        self.enc = BaseTransformer(embed_size, depth, hidden_dim,
                                   nhead=nhead, dropout=dropout, activation=activation, return_sides=return_sides)

        self.projection = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_size)
        )

    def forward(self, x):

        if self.return_sides:
            x, sides = self.enc(x)
            return self.projection(x), sides

        return self.projection(x)


class SubjectInputEncoder(nn.Module):
    def __init__(self, encoder, embed_size, n_subjects, sigma=1e-2):
        super().__init__()
        self.enc = encoder
        self.sigma = sigma
        self.subject_embedding = nn.Embedding(n_subjects, embed_size)

    def forward(self, x, subject):
        subject_embed = self.subject_embedding(subject)
        subject_embed = rearrange(subject_embed, 'b (c e) -> b c e', c=1)

        noise = torch.randn_like(subject_embed) * self.sigma
        subject_embed += noise

        x = torch.cat([subject_embed, x], dim=-2)

        return self.enc(x)


############################## Encoder Wrappers ##############################
class ClsOutEncoder(nn.Module):
    def __init__(self, enc):
        super().__init__()

        self.enc = enc
        self.cls_token = nn.Parameter(torch.randn(1, 1, enc.embed_size))

    def forward(self, x):
        cls_token = repeat(self.cls_token, '() c e -> b c e', b=x.size(0))
        x = torch.cat([cls_token, x], dim=1)

        out = self.enc(x)

        if len(out) == 2:  # if encoder also returns sides
            x, sides = out
            return x[:, 0], sides[:, :, 0]

        return out[:, 0]


class MultiTokenOutEncoder(nn.Module):
    def __init__(self, enc, n_tokens=4):
        super().__init__()

        self.enc = enc
        self.n_tokens = n_tokens
        self.out_tokens = nn.Parameter(
            torch.randn(1, n_tokens, enc.embed_size))

    def forward(self, x):
        tokens = repeat(self.out_tokens, '() c e -> b c e', b=x.size(0))
        x = torch.cat([tokens, x], dim=1)

        out = self.enc(x)

        if len(out) == 2:  # if encoder also returns sides
            x, sides = out
            return x[:, :self.n_tokens], sides[:, :, :self.n_tokens]

        out = out[:, :self.n_tokens]
        return out


class MeanOutEncoder(nn.Module):
    def __init__(self, enc):
        super().__init__()

        self.enc = enc

    def forward(self, x):
        out = self.enc(x)

        if len(out) == 2:  # if encoder also returns sides
            x, sides = out
            return torch.mean(x, dim=-2), torch.mean(sides, dim=-2)

        return torch.mean(out, dim=-2)


def get_encoder(encoder_info, embed_size):
    factories = {
        "fnet": FNet,
        "transformer1": BaseTransformer1,
        "transformer2": BaseTransformer2,
        "projection-transformer": TransformerWithProjection,
    }
    output_reduction_factories = {
        "mean": MeanOutEncoder,
        "cls": ClsOutEncoder
    }

    encoder_name = encoder_info["type"]
    output_reduction_name = encoder_info["output_reduction"]

    params = encoder_info["parameters"]
    hidden_dim = embed_size*params['hidden_factor']
    # output_reduction?

    if encoder_name in factories:
        enc = factories[encoder_name](
            embed_size, params["depth"], hidden_dim, dropout=params["dropout"])
        if output_reduction_name is not None:
            enc = output_reduction_factories[output_reduction_name](enc)
        return enc
    print(f"Unknown encoder option: {encoder_name}.")
