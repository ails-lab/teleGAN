import torch
import torch.nn as nn


class GLU(nn.Module):
    """GLU convolutional block."""

    def __init__(self):
        """Initialize GLU block."""
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# G networks
def upBlock(in_planes, out_planes):
    """Upsale the spatial size by a factor of 2."""
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


def Block3x3_relu(in_planes, out_planes):
    """3x3 convolution with ReLU while keeping the spatial size."""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    """Residual Block.

    The residual block consists of the following:
        - 3x3 Convolution with padding
        - 2D Batch Normalization
        - GLU block
        - 3x3 Convolution with padding
        - 2D Batch Normalization
    """

    def __init__(self, channel_num):
        """Initialize Residual Block."""
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


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
        self.relu = GLU()

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
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    """Generator's init stage class.

    Args:
    - ngf (int): dimension of the generators filters
    - nef (int): condition dimension
    - nz (int): noise dimension
    """

    def __init__(self, ngf, nef, nz):
        """Initialize the Generator's init stage."""
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = nz + nef

        self.define_module()

    def define_module(self):
        """Define the Init Stage generator module."""
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code=None):
        if c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)
        else:
            in_code = z_code
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    """Next stage class of the Generator.

    Args:
        - ngf (int): dimension of the generator's filters
        - nef (int): condition dimension
        - nz (int): noise dimension
        - num_residual (int, optional): number of residual blocks
    """

    def __init__(self, ngf, nef, nz, num_residual=2):
        """Initialize the next stage of the Generator."""
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef

        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        """Create a sequential model of <num_residual> <blocks>."""
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        """Define the Next Stage generator module."""
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code


class GET_IMAGE_G(nn.Module):
    """Last convolution to retrieve image.

    Args:
        - ngf (int): dimension of the generator's filters.
    """

    def __init__(self, ngf):
        """Initialize GET_IMAGE convolution."""
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    """Generator's network class.

    Args:
        - ngf (int): dimension of the generators filters
        - nef (int): condition dimension
        - nz (int): noise dimension
        - text_dim (int): text dimension
        - branch_num (int, optional): number of branches
    """

    def __init__(self, ngf, nef, nz, text_dim, branch_num=3):
        """Initialize the Generator."""
        super(G_NET, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.nz = nz
        self.text_dim = text_dim
        self.branch_num = branch_num
        self.define_module()

    def define_module(self):
        """Define the generator module."""
        self.ca_net = CA_NET(self.text_dim, self.ef_dim)

        if self.branch_num > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, self.ef_dim,
                                       self.nz)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if self.branch_num > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim, self.ef_dim,
                                       self.nz)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if self.branch_num > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2, self.ef_dim,
                                       self.nz)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)

    def forward(self, z_code, text_embedding=None):
        if text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None
        fake_imgs = []
        if self.branch_num > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if self.branch_num > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if self.branch_num > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs, mu, logvar


# D networks
def Block3x3_leakRelu(in_planes, out_planes):
    """3x3 Convolution with normalization and LeakyRelu."""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def downBlock(in_planes, out_planes):
    """Downsample the spatial size by a factor of 2."""
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def encode_image_by_16times(ndf):
    """Downsample the spatial size by a factor of 16."""
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


class D_NET64(nn.Module):
    """Discriminator's network for 64x64 images.

    Args:
        - ndf (int): dimension of the discriminators filters
        - nef (int): condition dimension
    """

    def __init__(self, ndf, nef):
        """Initialize the 64x64 Discriminator."""
        super(D_NET64, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.define_module()

    def define_module(self):
        """Define the 64x64 discriminator module."""
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)

        if c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]


class D_NET128(nn.Module):
    """Discriminator's network for 128x128 images.

    Args:
        - ndf (int): dimension of the discriminators filters
        - nef (int): condition dimension
    """

    def __init__(self, ndf, nef):
        """Initialize the 128x128 Discriminator."""
        super(D_NET128, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.define_module()

    def define_module(self):
        """Define the 128x128 discriminator module."""
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        if c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]


class D_NET256(nn.Module):
    """Discriminator's network for 256x256 images.

    Args:
        - ndf (int): dimension of the discriminators filters
        - nef (int): condition dimension
    """

    def __init__(self, ndf, nef):
        """Initialize the 256x256 Discriminator."""
        super(D_NET256, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.define_module()

    def define_module(self):
        """Define the 256x256 discriminator module."""
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        if c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ndf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ndf*8 x 4 x 4
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]
