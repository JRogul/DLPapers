import torch
import torch.nn as nn

class EncoderConvBLock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3, stride =1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=3, stride =1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):

        out = self.conv_layers(x)
        return out


class DecoderConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3, stride =1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=3, stride =1, padding=0),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=2, stride =2)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_list = nn.ModuleList([EncoderConvBLock(1, 64), 
                                       EncoderConvBLock(64, 128), 
                                       EncoderConvBLock(128, 256), 
                                       EncoderConvBLock(256, 512)])

        self.dec_list = nn.ModuleList([DecoderConvBlock(512, 1024), 
                                       DecoderConvBlock(1024, 512), 
                                       DecoderConvBlock(512, 256), 
                                       DecoderConvBlock(256, 128)])
        
        self.final_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride =1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride =1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride =1, padding=0)
        )
    
    def forward(self, x):

        out = x
        for enc_block in self.enc_list:
            out = enc_block(out)

        for dec_block in self.dec_list:
            out = dec_block(out)

        out = self.final_block(out)
        return out
    
model = Unet()
print(model)
#outputs 6 2 388 388
print(model(torch.randn(6, 1, 572, 572)).shape)