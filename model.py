import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = torch.nn.functional.relu(self.conv1(x))
        y = self.conv2(y)
        return torch.nn.functional.relu(y+x)

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=9,padding=4),
            # nn.Upsample(scale_factor=4,mode= 'bicubic',align_corners=True),
            nn.BatchNorm2d(64,0.8,0.1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,4,4),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64, 0.8, 0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64, 0.8, 0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64, 0.8, 0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64, 0.8, 0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64, 0.8, 0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64, 0.8, 0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64, 0.8, 0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            ResidualBlock(64),
            nn.BatchNorm2d(64,0.8,0.1),
            nn.Conv2d(64,50,1,1),
            nn.BatchNorm2d(50,0.8,0.1),
            nn.Conv2d(50, 1, kernel_size=9, padding=4),
            nn.Tanh()


        )


    def forward(self, x):
        out1 = self.conv1(x)
        return out1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),


            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),


            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),


            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),


            nn.Conv2d(32, 1,kernel_size=3, padding=1),
            nn.Sigmoid()


        )
    def forward(self,x):
        x = self.model(x)


        return x



