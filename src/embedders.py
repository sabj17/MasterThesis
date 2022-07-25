import torch
import torch.nn as nn
from einops import reduce, rearrange, repeat
import torch.nn.functional as F
from tokenizers import *
import math


class PositionalEmbedding1D(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=1000):
        super(PositionalEmbedding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_weights = torch.empty(1, max_len, dim)
        nn.init.normal_(pos_weights, std=0.02)
        self.pos_embedding = nn.Parameter(pos_weights)

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1)]
        return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class PatchEmbeddings(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, embed_size):
        super().__init__()

        self.to_token = EEGtoPatchTokens(patch_size)
        _, n_tokens, token_size = self.to_token(
            torch.randn(1, n_channels, seq_len)).shape

        self.linear_embedding = nn.Linear(token_size, embed_size)

    def forward(self, x):
        x = self.to_token(x)
        x = self.linear_embedding(x)
        return x


class ChannelPatchEmbeddings(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, embed_size):
        super().__init__()
        assert embed_size % patch_size == 0

        n_components = embed_size // patch_size
        self.to_token = EEGtoPatchTokens(patch_size)
        self.linear_embedding = nn.Linear(n_channels, n_components, bias=False)

    def forward(self, x):
        x = rearrange(x, 'b c t -> b t c')
        x = self.linear_embedding(x)
        x = rearrange(x, 'b t c -> b c t')

        x = self.to_token(x)
        return x


class ChannelAverageEmbeddings(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, embed_size):
        super().__init__()

        self.patch_size = patch_size
        self.to_token = EEGtoPatchTokens(patch_size)
        self.linear_embedding = nn.Linear(n_channels, embed_size, bias=False)

    def forward(self, x):
        x = self.to_token(x)
        x = rearrange(x, 'b s (c p) -> b s p c', p=self.patch_size)
        x = self.linear_embedding(x)
        x = reduce(x, 'b s p c -> b s c', 'mean')

        return x


class STPatchEmbeddings(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, embed_size):
        super().__init__()

        self.to_token = EEGtoPatchTokens(patch_size)

        n_components = int(math.sqrt(embed_size))
        print(n_components)
        self.channel_embed = nn.Linear(n_channels, n_components)
        self.time_embed = nn.Linear(patch_size, n_components)

    def forward(self, x):
        x = self.to_token(x)

        x = rearrange(x, 'b s (c p) ->  b s c p', p=5)
        x = self.time_embed(x)
        x = rearrange(x, 'b s c p ->  b s p c')
        x = self.channel_embed(x)
        x = rearrange(x, 'b s p c ->  b s (p c)')

        return x


class EmbeddingWithSubject(nn.Module):
    def __init__(self, embedder, embed_size, n_subjects):
        super().__init__()
        self.embedder = embedder
        self.subject_embed = nn.Embedding(n_subjects, embed_size)

    def forward(self, x, subject):
        x = self.embedder(x)

        subject_token = self.subject_embed(subject)
        subject_token = rearrange(subject_token, 'b e -> b 1 e')
        x = torch.cat([subject_token, x], dim=-2)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels, seq_len, n_components, inter=30):
        super().__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        # You could choose to do that for less computation
        self.extract_sequence = int(self.sequence_num / self.inter)

        self.query = nn.Sequential(
            nn.Linear(n_channels, n_components),
            # also may introduce improvement to a certain extent
            nn.LayerNorm(n_components),
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(n_channels, n_components),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_components),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(n_channels, n_components),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_components),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(
            1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = rearrange(x, 'b (o c) s -> b o c s', o=1)

        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        '''
        temp = self.pooling(x)

        temp = rearrange(temp, 'b o c s->b o s c')
        channel_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        channel_key = rearrange(self.key(temp), 'b o s c -> b o c s')
        '''

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum(
            'b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s',
                           x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b (o c) s')
        return out


class ChannelAttentionEmbedding(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, embed_size, n_components=32):
        super().__init__()

        self.spatial_attn = nn.Sequential(
            nn.LayerNorm(seq_len),
            ChannelAttention(n_channels, seq_len, n_components),
            nn.Dropout(0.5),
        )
        self.to_token = EEGtoPatchTokens(patch_size)

    def forward(self, x):
        x = x + self.spatial_attn(x)
        x = self.to_token(x)

        return x


class Conv1dEmbedding(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, embed_size, depth=3):
        super().__init__()

        layer_list = []
        for i in range(depth):
            kernel_stride = 3 if i == 0 else 2
            kernel_stride = 1 if i == depth-1 else kernel_stride
            infeat = n_channels if i == 0 else embed_size

            conv1d = nn.Sequential(
                nn.Conv1d(infeat, embed_size, kernel_stride,
                          stride=kernel_stride),
                nn.Dropout2d(0.5),
                nn.BatchNorm1d(embed_size),
                nn.GELU(),
            )
            layer_list.append(conv1d)

        self.conv_embed = nn.Sequential(
            *layer_list
        )

    def forward(self, x):
        x = self.conv_embed(x)
        x = rearrange(x, 'b e c -> b c e')
        return x


class CSPLikeEmbedding(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, embed_size, n_components=32):
        super().__init__()

        self.spatial_filtering = nn.Conv2d(
            1, n_components, (n_channels, 1), bias=False)

        self.to_token = EEGtoPatchTokens(patch_size)

    def forward(self, x):
        x = rearrange(x, 'b (o c) s -> b o c s', o=1)
        x = self.spatial_filtering(x)
        x = x**2
        x = torch.log(x)
        x = rearrange(x, 'b o c s -> b (o c) s')
        x = self.to_token(x)
        return x


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, n_channels, n_slices, embed_size):
        super(SpatioTemporalEmbedding, self).__init__()

        self.spatial_encoding = nn.Parameter(
            torch.randn(1, n_channels, 1, embed_size))
        self.temporal_encoding = nn.Parameter(
            torch.randn(1, 1, n_slices, embed_size))
        self.n_channels = n_channels
        self.n_slices = n_slices
        self.embed_size = embed_size

    def forward(self, x):
        x = x.view(-1, self.n_channels, self.n_slices, self.embed_size)
        x = x + self.spatial_encoding
        x = x + self.temporal_encoding
        x = x.view(-1, self.n_channels * self.n_slices, self.embed_size)

        return x


def get_embedder(embedder_info, data_exp, embed_size):
    factories = {
        "channelembed": ChannelPatchEmbeddings,
        "channelembed2": ChannelAverageEmbeddings,
        "embedder1": PatchEmbeddings,
        "convembed": Conv1dEmbedding,
        "csp": CSPLikeEmbedding,
        "channel_attention": ChannelAttentionEmbedding,
        "stembed": STPatchEmbeddings
    }

    embedder_name = embedder_info['type']
    params = embedder_info['parameters']

    n_channels = data_exp['tensor'].shape[1]
    seq_len = data_exp['tensor'].shape[2]
    patch_size = params['patch_size']

    if embedder_name in factories:
        return factories[embedder_name](n_channels, seq_len, patch_size, embed_size)
    print(f"Unknown embedding option: {embedder_name}.")


''' For EEG transformer
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (22, 5), stride=(1, 5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # position
        # x += self.positions
        return x
'''
