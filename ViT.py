import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim  = dim

        self.norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(nn.Linear(self.dim, self.dim * 4 ),
                                nn.GELU(),
                                nn.Linear(self.dim * 4, self.dim))
    def forward(self, x):

        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Linear(dim, dim)
        self.keys = nn.Linear(dim, dim)
        self.values = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        out =  self.norm(x)
        
        q = self.queries(out)
        k = self.keys(out)
        v = self.values(out)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        att = self.softmax((q @ k.transpose(2, 3)) / torch.sqrt(torch.tensor(self.dim_head))) @ v
        att = rearrange(att, 'b h n d -> b n (h d)', h=self.heads)
        return att


class ViT(nn.Module):

    def __init__ (self, patch_size, dim, img_shapes=(224, 224)):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        
        self.height, self.width = img_shapes
        self.N = int(self.height / self.patch_size * self.width / self.patch_size)
        self.dim = dim
        self.ppc = self.patch_size ** 2 * 3
        
        self.img_embeddings = Rearrange('b c (h p1) (w p2) ->b (h w) (c p1 p2)', p1 = self.patch_size, p2 = self.patch_size)
        self.linear_layer = nn.Linear(self.ppc, self.dim)

    def forward(self, x):
        print(f'imput shape shape:{x.shape}')
        out = self.img_embeddings(x) # B x C x H x W ---> Bx N × (P*P·C)
        #out = out.transpose(1, 2).unsqueeze(1)
        print(out.shape)
        out = self.linear_layer(out) # Bx N × (P*P·C) @ P*P*C X N ---> N x N
        print(out.shape)
        pos_embeddings = torch.randn(self.N + 1, self.ppc)
        print(f'positional_embeddings shape:{pos_embeddings.shape}')
        return out