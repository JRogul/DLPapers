import torch
import numpy as np
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, 
                      kernel_size=11, stride=4, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, 
                      kernel_size=5, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),            
            nn.Conv2d(in_channels=256, out_channels=384, 
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.linear = nn.Sequential(
            nn.Linear(9216, 4096, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 4096, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 1000, bias=False)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out
