"""This module implements the networks used in TeleGAN."""
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms


def conv3x3(in_planes, out_planes, bias=False):
    """3x3 convolution with padding=1."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=bias)


def conv4x4(in_planes, out_planes, bias=True):
    """4x4 convolution with stride=2 and padding=1."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2,
                     padding=1, bias=bias)


# G networks
class CA_NET(nn.Module):
    """Conditioning Augmentation network.

    Args:
        - t_dim (int): text dimension
        - ef_dim (int): embedding dimension
    """

    def __init__(self, t_dim, ef_dim):
        """Initialize the conditioning augmentation."""
        super(CA_NET, self).__init__()
        self.t_dim = t_dim
        self.ef_dim = ef_dim
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = nn.GLU(1)

    def encode(self, text_embedding):
        """Encode the text embedding."""
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        """Reparametrize the encoded text embeddings."""
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        """Forward propagation."""
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class GET_IMAGE_G(nn.Module):
    """Last convolution to retrieve image.

    Args:
        - ngf (int): dimension of the generator's filters
        - nc (int): number of channels
    """

    def __init__(self, ngf, nc):
        """Initialize GET_IMAGE convolution."""
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.nc = nc
        self.img = nn.Sequential(
            conv3x3(ngf, self.nc),
            nn.Tanh()
        )

    def forward(self, h_code):
        """Forward propagation."""
        out_img = self.img(h_code)
        return out_img


class BAW(nn.Module):
    """Class for the black and white stage.

    Args:
    - ngf (int): dimension of the generators filters
    - nef (int): condition dimension
    - nz (int): noise dimension
    """

    def __init__(self, ngf, nef, nz):
        """Initialize the Generator's init stage."""
        super(BAW, self).__init__()
        self.gf_dim = ngf
        self.in_dim = nz + nef

        self.define_module()

    def upBlock(self, in_planes, out_planes):
        """Upscale the spatial size by a factor of 2."""
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes * 2),
            nn.BatchNorm2d(out_planes * 2),
            nn.GLU(1)
        )
        return block

    def define_module(self):
        """Define the Black and White generator module."""
        in_dim = self.in_dim
        ngf = self.gf_dim

        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            nn.GLU(1))

        self.upsample1 = self.upBlock(ngf, ngf // 2)
        self.upsample2 = self.upBlock(ngf // 2, ngf // 4)
        self.upsample3 = self.upBlock(ngf // 4, ngf // 8)
        self.upsample4 = self.upBlock(ngf // 8, ngf // 16)
        self.upsample5 = self.upBlock(ngf // 16, ngf // 32)

        self.img = GET_IMAGE_G(ngf // 32, 1)

    def forward(self, z_code, c_code=None):
        """Forward propagation."""
        if c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)
        else:
            in_code = z_code
        # state size 32ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 16ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 8ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 4ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size 2ngf x 64 x 64
        out_code = self.upsample4(out_code)
        # state size ngf x 128 x 128
        out_code = self.upsample5(out_code)
        # state size 3 x 128 x 128
        fake_img = self.img(out_code)

        return fake_img


class ENCOLOR(nn.Module):
    """Class for the colorization stage.

    Args:
    - ngf (int): dimension of the generators filters
    - nef (int): condition dimension
    """

    def __init__(self, ngf, nef):
        """Initialize the ENCOLOR stage."""
        super(ENCOLOR, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.in_planes = ngf + nef

        self.define_module()

    def doubleConv(self, in_planes, out_planes, mid_planes=None):
        """Double convolution layer of UNET.

        Construct a convolutional unit comprising of two conv layers
        followed by a batch normalization layer and Leaky ReLU.
        """
        if not mid_planes:
            mid_planes = out_planes

        block = nn.Sequential(
            conv3x3(in_planes, mid_planes),
            nn.BatchNorm2d(mid_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(mid_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    def DownScale(self, in_planes, out_planes):
        """Downscaling with maxpool and double convolution."""
        return nn.Sequential(
            nn.MaxPool2d(2, 2),
            self.doubleConv(in_planes, out_planes)
        )

    def UpScale(self, in_planes, out_planes):
        """Upscaling with upsample and double convolution."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.doubleConv(in_planes, out_planes, in_planes // 2)
        )

    def define_module(self):
        """Define the ENCOLOR generator module."""
        self.down_0 = self.doubleConv(1, 32)
        self.down_1 = self.DownScale(32, 64)
        self.down_2 = self.DownScale(64, 128)
        self.down_3 = self.DownScale(128, 256)
        self.down_4 = self.DownScale(256, 512)

        self.upsample = \
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = self.doubleConv(512 + 256 + self.ef_dim, 256)
        self.up_3 = self.doubleConv(256 + 128 + self.ef_dim, 128)
        self.up_2 = self.doubleConv(128 + 64 + self.ef_dim, 64)
        self.up_1 = self.doubleConv(64 + 32 + self.ef_dim, 32)

        self.img = GET_IMAGE_G(32, 3)

    def forward(self, x, c_code):
        """Forward propagation."""
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code16 = c_code.repeat(1, 1, 16, 16)
        c_code32 = c_code.repeat(1, 1, 32, 32)
        c_code64 = c_code.repeat(1, 1, 64, 64)
        c_code128 = c_code.repeat(1, 1, 128, 128)

        # Input size: 1 x 128 x 128
        x0d = self.down_0(x)
        # state size 32 x 128 x 128
        x1d = self.down_1(x0d)
        # state size 64 x 64 x 64
        x2d = self.down_2(x1d)
        # state size 128 x 32 x 32
        x3d = self.down_3(x2d)
        # state size 256 x 16 x 16
        x4d = self.down_4(x3d)
        # state size 512 x 8 x 8

        x4u = torch.cat([x3d, self.upsample(x4d), c_code16], 1)
        # state size (256+512+nef) x 16 x 16
        x3u = self.up_4(x4u)
        # state size 256 x 16 x 16
        x2u = self.up_3(torch.cat([x2d, self.upsample(x3u), c_code32], 1))
        # state size 128 x 32 x 32
        x1u = self.up_2(torch.cat([x1d, self.upsample(x2u), c_code64], 1))
        # state size 64 x 64 x 64
        x0u = self.up_1(torch.cat([x0d, self.upsample(x1u), c_code128], 1))
        # state size 32 x 128 x 128

        fake_img = self.img(x0u)
        # state size 3 x 128 x 128
        return fake_img


class ResBlock(nn.Module):
    """Residual Block.

    The residual block consists of the following:
        - 3x3 Convolution with padding
        - 2D Batch Normalization
        - PReLU
        - 3x3 Convolution with padding
        - 2D Batch Normalization
        - ReLU
    """

    def __init__(self, channels):
        """Initialize a Residual Block."""
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward propagation."""
        residual = x
        out = self.block(x)
        out += residual
        return out


class UpsampleBLock(nn.Module):
    """Upsample block as implemented in SRGAN."""

    def __init__(self, in_channels, up_scale):
        """Initialize Upsample block."""
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2,
                              kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        """Forward propagation."""
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class SuperRes(nn.Module):
    """Super resolution stage class.

    Args:
    - ngf (int): dimension of the generators filters
    - n_residuals (int, optional): number of residual blocks
        used (Default: 5)
    - nc (int, optional): number of channels (Default: 3)
    """

    def __init__(self, ngf, n_residuals=5, nc=3):
        """Initialize the Super Resolution stage."""
        super(SuperRes, self).__init__()
        self.gf_dim = ngf
        self.n_residuals = n_residuals
        self.nc = nc

        self.define_module()

    def _make_layer(self, block, in_channels):
        """Create a sequential model of <n_residuals> <block>."""
        layers = []
        for i in range(self.n_residuals):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def define_module(self):
        """Define the Super Resolution generator module."""
        ngf = self.gf_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, ngf, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.residual = self._make_layer(ResBlock, ngf)
        self.block = nn.Sequential(
            conv3x3(ngf, ngf),
            nn.BatchNorm2d(ngf)
        )

        self.upsample = nn.Sequential(
            UpsampleBLock(ngf, 2),
            nn.Conv2d(ngf, self.nc, kernel_size=9, padding=4)
        )

    def forward(self, x):
        """Forward propagation."""
        # Encode img
        x = self.encoder(x)
        # Size ngf x 128 x 128

        # Input size: ngf x 128 x 128
        output = self.residual(x)
        # State size: ngf x 128 x 128
        output = self.block(output)
        # State size: ngf x 128 x 128
        output = self.upsample(output + x)
        # State size: 3 x 256 x 256

        return (torch.tanh(output) + 1) / 2


class Generator(nn.Module):
    """Network Generator class.

    Args:
    - stage (int): generator's stage (1,2 or 3)
    - ngf (int): dimension of the generators filters
    - nef (int): condition dimension
    - nz (int): noise dimension
    - text_dim (int): original embedding dimension
    """

    def __init__(self, stage, ngf, nef, nz, text_dim):
        """Initialize the Generator's init stage."""
        super(Generator, self).__init__()
        self.stage = stage
        self.gf_dim = ngf
        self.ef_dim = nef
        self.nz = nz
        self.text_dim = text_dim

        self.define_module()

    def define_module(self):
        """Define the generator module."""
        self.ca_net = CA_NET(self.text_dim, self.ef_dim)

        # Black and white stage
        self.baw_net = BAW(self.gf_dim * 32, self.ef_dim, self.nz)

        if self.stage > 1:
            # Colorization stage
            for p in self.baw_net.parameters():
                p.requires_grad = False
            self.encolor_net = ENCOLOR(self.gf_dim, self.ef_dim)

        if self.stage > 2:
            # Super Resolution stage
            for p in self.encolor_net.parameters():
                p.requires_grad = False
            self.super_res = SuperRes(self.gf_dim, n_residuals=5)

    def forward(self, z_code, text_embedding=None, train=True):
        """Forward propagation."""
        fake_imgs = []
        if self.stage == 3 and train:
            fake_SR = self.super_res(z_code)
            fake_imgs.append(fake_SR)
            return fake_imgs, None, None

        if text_embedding is not None:
            c_code_baw, mu, logvar = self.ca_net(text_embedding)
            c_code_encolor, mu_encolor, logvar_encolor = \
                self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None

        # First stage
        fake_baw = self.baw_net(z_code, c_code_baw)
        fake_imgs.append(fake_baw)

        if self.stage > 1:
            # Second stage
            fake_colored = self.encolor_net(fake_baw.detach(), c_code_encolor)
            fake_imgs.append(fake_colored)
            if mu is not None:
                mu = mu_encolor
                logvar = logvar_encolor

        if self.stage > 2:
            # Third stage
            fake_SR = self.super_res(fake_colored.detach())
            fake_imgs.append(fake_SR)

        return fake_imgs, mu, logvar


class DNET_BAW(nn.Module):
    """Discriminator class for the Black and White stage.

    Args:
        - ndf (int, optional): dimension of Black and White
            discriminator's filters. (Default: 64)
        - nef (int, optional): Projected embeddings dimensions.
            (Default: 128)
    """

    def __init__(self, ndf=64, nef=128):
        """Initialize the Black and White stage discriminator.

        Other attributes:
            - nc (int): Number of channels.
        """
        super(DNET_BAW, self).__init__()
        self.nc = 1
        self.ef_dim = nef
        self.df_dim = ndf

        self.define_module()

    def define_module(self):
        """Define the ENCOLOR discriminator module."""
        ndf = self.df_dim
        self.netD_1 = nn.Sequential(
            # input is (nc) x 128 x 128
            conv4x4(self.nc, ndf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # state size: (ndf) x 64 x64
            conv4x4(ndf, ndf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # state size: (ndf) x 32 x 32
            conv4x4(ndf, ndf * 2),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # state size: (ndf*2) x 16 x 16
            conv4x4(ndf * 2, ndf * 4),
            nn.BatchNorm2d(num_features=ndf * 4),

            # state size: (ndf*4) x 8 x 8
            conv4x4(ndf * 4, ndf * 8),
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
                bias=True
            ),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # state size: (ndf*2) x 4 x 4
            conv3x3(ndf * 2, ndf * 2, bias=True),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # state size: (ndf*2) x 4 x 4
            conv3x3(ndf * 2, ndf * 8, bias=True),
            nn.BatchNorm2d(num_features=ndf * 8)
        )
        self.lrelu = nn.Sequential(
            # state size: (ndf*8) x 4 x 4
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.netD_3 = nn.Sequential(
            # state size: (ndf*8 + nef) x 4 x 4
            nn.Conv2d(
                in_channels=ndf * 8 + self.ef_dim,
                out_channels=ndf * 8,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.netD_4 = nn.Sequential(
            # state size: ndf*8 x 4 x 4
            nn.Conv2d(
                in_channels=ndf * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True
            ),
            # state size: 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x, c_code=None):
        """Forward propagation."""
        output_1 = self.netD_1(x)
        output_2 = self.netD_2(output_1)
        output_3 = self.lrelu(output_1 + output_2)

        if c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c = torch.cat((c_code, output_3), 1)
            h_c = self.netD_3(h_c)
        else:
            h_c = output_3

        output = self.netD_4(h_c)
        uncond_output = self.netD_4(output_3)

        return [output.view(-1), uncond_output.view(-1)]


class DNET_COLORED(nn.Module):
    """Discriminator class for the ENCOLOR stage.

    Args:
        - ndf (int, optional): Number of discriminator filters in the
            first convolutional layer. (Default: 64)
        - nef (int, optional): Projected embeddings dimensions. (Default: 128)
    """

    def __init__(self, ndf=64, nef=128):
        """Initialize the ENCOLOR stage discriminator.

        Other attributes:
            - nc (int): Number of channels.
        """
        super(DNET_COLORED, self).__init__()
        self.nc = 3
        self.ef_dim = nef
        self.df_dim = ndf

        self.define_module()

    def define_module(self):
        """Define stage 2 discriminator's model."""
        nef = self.ef_dim
        ndf = self.df_dim

        self.netD_1 = nn.Sequential(
            # Input size nc x 128 x 128
            conv4x4(self.nc, ndf),
            nn.BatchNorm2d(num_features=ndf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # State size ndf x 64 x 64
            conv4x4(ndf, ndf * 2),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # State size ndf*2 x 32 x 32
            conv4x4(ndf * 2, ndf * 4),
            nn.BatchNorm2d(num_features=ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # State size ndf*4 x 16 x 16
            conv4x4(ndf * 4, ndf * 8),
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # State size ndf*8 x 8 x 8
            conv4x4(ndf * 8, ndf * 16),
            nn.BatchNorm2d(num_features=ndf * 16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State size ndf*16 x 4 x 4
        )

        self.netD_2 = nn.Sequential(
                # state size: (ndf*16 + nef) x 4 x 4
                nn.Conv2d(
                    in_channels=ndf * 16 + self.ef_dim,
                    out_channels=ndf * 16,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                ),
                nn.BatchNorm2d(num_features=ndf * 16),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.netD_3 = nn.Sequential(
                # state size: ndf*16 x 4 x 4
                nn.Conv2d(
                    in_channels=ndf * 16,
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=True
                ),
                # state size: 1 x 1 x 1
                nn.Sigmoid()
            )

    def forward(self, x, c_code=None):
        """Forward propagation."""
        x = self.netD_1(x)

        if c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c = torch.cat((c_code, x), 1)
            h_c = self.netD_2(h_c)
        else:
            h_c = x

        output = self.netD_3(h_c)
        uncond_output = self.netD_3(x)

        return [output.view(-1), uncond_output.view(-1)]


class DNET_SR(nn.Module):
    """Discriminator class for the Super Resolution stage.

    Args:
        - ndf (int, optional): Number of discriminator filters in the
            first convolutional layer. (Default: 64)
    """

    def __init__(self, ndf=64):
        """Initialize the ENCOLOR stage discriminator.

        Other attributes:
            - nc (int): Number of channels.
        """
        super(DNET_SR, self).__init__()
        self.nc = 3
        self.df_dim = ndf

        self.define_module()

    def downBlock(self, in_planes, out_planes):
        """Downsample the spatial size by a factor of 2."""
        block = nn.Sequential(
            conv3x3(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv4x4(out_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    def define_module(self):
        """Define stage 3 discriminator's model."""
        ndf = self.df_dim

        self.net = nn.Sequential(
            self.downBlock(self.nc, ndf),
            self.downBlock(ndf, ndf * 2),
            self.downBlock(ndf * 2, ndf * 4),
            self.downBlock(ndf * 4, ndf * 8),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 16, 1, kernel_size=1)
        )

    def forward(self, x, c_code=None):
        """Forward propagation."""
        batch_size = x.size(0)
        # Input size: 3 x 256 x 256
        output = self.net(x)

        return torch.sigmoid(output.view(batch_size))
