import torch
import torch.nn as nn
import torchvision
import numpy as np

class VGG_Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.layers = nn.ModuleList(nn.Conv2d(self.out_channels,self.out_channels, kernel_size=3, 
                                    stride=1, padding=1) for _ in range(self.num_layers -1))
        
        self.layers.append(nn.MaxPool2d(2, 2))

    def forward(self, x):
        out = self.first_layer(x)
        for layer in self.layers:
            out = layer(out)
        return out


class VGG(nn.Module):
    def __init__(self, first_block_size, blocks_sizes = []):
        super().__init__()
        self.first_block_size = first_block_size
        self.blocks_sizes = blocks_sizes

        if self.first_block_size == 1:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
                )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
                )

        self.block1 = VGG_Block(blocks_sizes[0], 64, 128)
        self.block2 = VGG_Block(blocks_sizes[1], 128, 256)
        self.block3 = VGG_Block(blocks_sizes[2], 256, 512)
        self.block4 = VGG_Block(blocks_sizes[3], 512, 512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7,7))

        self.linear = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        out = self.layer(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out


model = torchvision.models.vgg19()
print(model)

model2 = VGG(2, [2,4,4,4])
print(model2)
print(model2(torch.randn(10,3,224,224)).shape)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

model_parameters2 = filter(lambda p: p.requires_grad, model2.parameters())
params2 = sum([np.prod(p.size()) for p in model_parameters2])
print(params, params2)

