import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


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