"""This module implements functions used by TeleGAN."""
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.nn import init

import os
import errno
import numpy as np
from PIL import Image
from copy import deepcopy
from matplotlib import pyplot as plt


def save_images(images, path, stage, epoch=-1):
    """Save a single or a batch of images.

    This method takes as input a tensor containing the images to
    be saved, inverses the normalization and saves the images in
    the directory specified while initializing the model. The
    directory will have to following format:

    -- images
        -- real-images
        -- epoch-[#]
        -- other_images (if this method is called outside of training)

    Args:
        - images (tensor): A tensor containing the images to be
            saved. The tensor must have the following format:
                    (number_of_images x C x H x W)
        - path (string): Directory's path to save the images.
        - epoch (int, optional): The current epoch. It is used for
            naming purposes. If not given as input, the images will
            be saved inside the 'other_images' directory.
    """
    if epoch == -1:
        loc = os.path.join(
            path,
            'other-images'
        )
    elif epoch == 0:
        loc = os.path.join(
            path,
            'real-images'
        )
    else:
        loc = os.path.join(
            path,
            f'epoch-{epoch}'
        )

    if not os.path.exists(loc):
        os.makedirs(loc)

    if images.size(1) == 1:
        inverse_normalize = transforms.Normalize(
            mean=[-1],
            std=[2]
        )
    else:
        inverse_normalize = transforms.Normalize(
            mean=[-1, -1, -1],
            std=[2, 2, 2]
        )

    for i, img in enumerate(images):
        # Inverse the normalization and transpose the shape
        # to [HEIGHT, WIDTH, CHANNELS]
        if stage != 3:
            img = inverse_normalize(img)
        img = torch.squeeze(img.permute(1, 2, 0))
        img = (img.cpu().numpy() * 255).astype(np.uint8)

        im = Image.fromarray(img)
        im.save(os.path.join(loc, f'image-{i}.png'))


def save_checkpoints(netG, netD, G_losses, D_losses, epoch, path):
    """Save Generator and Discriminator states along with a plot.

    This method saves the Generator and Discriminator states and a plot
    of their total losses (until the current batch) to the following files:
        - generator.pkl
        - discriminator.pkl
        - losses.png
    inside of the [path to checkpoints directory]/epoch-[#]/ directory.

    Args:
        - netG (nn.Module): The generator object to save.
        - netsD (nn.Module): A list of the discriminator objects to save.
        - G_losses (list): List of current generator's losses.
        - D_losses (list): List of current discriminator's losses.
        - epoch (int): The current epoch. It is used for
            naming purposes.
        - path (string): The directory's path to save the checkpoints.
    """
    loc = os.path.join(
            path,
            f'epoch-{epoch}'
    )
    if not os.path.exists(loc):
        os.makedirs(loc)

    torch.save(
        netG.module.state_dict(),
        os.path.join(loc, 'generator.pkl')
    )

    torch.save(
        netD.module.state_dict(),
        os.path.join(loc, f'discriminator.pkl')
    )

    plt.figure()
    plt.grid()
    x = np.arange(len(G_losses))
    y_G = G_losses
    y_D = D_losses
    plt.plot(x, y_G, 'b', label='Generator losses')
    plt.plot(x, y_D, 'r', label='Discriminator losses')
    plt.legend(loc="upper right")
    plt.title('Generator and Discriminator losses')
    plt.xlabel('Number of training batches')
    plt.ylabel('Loss')
    plt.savefig(
        os.path.join(loc, 'losses.png')
    )


def copy_G_params(model):
    """Copy the paramaters of a generator's model."""
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def KL_loss(mu, logvar):
    """Compute the Kullback-Leibler divergence loss."""
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_mean_covariance(img):
    """Compute mean covariance for the color consistency loss."""
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance
