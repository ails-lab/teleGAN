import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    """Generator class of Text To Image Synthesis GAN."""

    def __init__(self, ne=1024, nt=128, nz=100, ngf=128):
        """Initialize the Generator.

        Args:
           - ne (int, optional): Original embeddings dimensions.
           - nt (int, optional): Projected embeddings dimensions.
           - nz (int, optional): Dimension of the noise input.
           - ngf (int, optional): Number of generator filters in the
                first convolutional layer.
        Other attributes:
            - nc (int): Number of channels.
        """
        super(Generator, self).__init__()
        self.nc = 3
        self.nt = nt

        # Projecting the 1024-dimensional embeddings
        # to 128 dimensions.
        self.projected = nn.Sequential(
            nn.Linear(in_features=ne, out_features=nt),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Main network
        self.netG_1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nz + nt,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf * 8)
        )
        self.netG_2 = nn.Sequential(
            # state size: (ngf*8) x 4 x 4
            nn.Conv2d(
                in_channels=ngf * 8,
                out_channels=ngf * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=ngf * 2,
                out_channels=ngf * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=ngf * 2,
                out_channels=ngf * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf * 8)
        )
        self.netG_3 = nn.Sequential(
            nn.ReLU(inplace=True),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels=ngf * 8,
                out_channels=ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf * 4)
        )
        self.netG_4 = nn.Sequential(
            # state size: (ngf*4) x 8 x 8
            nn.Conv2d(
                in_channels=ngf * 4,
                out_channels=ngf,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=ngf,
                out_channels=ngf,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=ngf,
                out_channels=ngf * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf * 4)
        )
        self.netG_5 = nn.Sequential(
            nn.ReLU(inplace=True),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=ngf * 4,
                out_channels=ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),

            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=ngf * 2,
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),

            # state size: (ngf) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=self.nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
            # state size: (nc) x 64 x 64
        )

    def forward(self, z, t):
        h = self.projected(t).view(-1, self.nt, 1, 1)
        zh = torch.cat([z, h], dim=1)
        output_1 = self.netG_1(zh)
        output_2 = self.netG_2(output_1)
        output_3 = self.netG_3(output_1 + output_2)
        output_4 = self.netG_4(output_3)
        output = self.netG_5(output_3 + output_4)

        return output


class Discriminator(nn.Module):
    """Discriminator class of Text To Image Synthesis GAN."""

    def __init__(self, ne=1024, nt=128, ndf=64):
        """Initialize the Discriminator.

        Args:
           - ne (int, optional): Original embeddings dimensions.
           - nt (int, optional): Projected embeddings dimensions.
           - ndf (int, optional): Number of discriminator filters in the
                first convolutional layer.
        Other attributes:
            - nc (int): Number of channels.
        """
        super(Discriminator, self).__init__()
        self.nc = 3
        self.nt = nt

        self.netD_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(
                in_channels=self.nc,
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # state size: (ndf) x 32 x 32
            nn.Conv2d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # state size: (ndf*2) x 16 x 16
            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf * 4),

            # state size: (ndf*4) x 8 x 8
            nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf * 8)
        )
        self.netD_2 = nn.Sequential(
            # state size: (ndf*8) x 4 x 4
            nn.Conv2d(
                in_channels=ndf * 8,
                out_channels=ndf * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf * 8)
        )
        self.lrelu = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # state size: (ndf*8) x 4 x 4
        )

        self.projected = nn.Sequential(
            nn.Linear(in_features=ne, out_features=nt),
            nn.BatchNorm1d(num_features=nt),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.netD_3 = nn.Sequential(
            # state size: (ndf*8 + nt) x 4 x 4
            nn.Conv2d(
                in_channels=ndf * 8 + nt,
                out_channels=ndf * 8,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=ndf * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        output_1 = self.netD_1(x)
        output_2 = self.netD_2(output_1)
        output_3 = self.lrelu(output_1 + output_2)
        h = self.projected(t).view(-1, self.nt, 1, 1).repeat(1, 1, 4, 4)
        xh = torch.cat([output_3, h], dim=1)
        output = self.netD_3(xh)

        return output.view(-1, 1).squeeze(1)
