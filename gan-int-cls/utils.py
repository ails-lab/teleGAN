import torch
import torch.nn as nn
import torchvision.transforms as transforms

import os
import errno
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt


def save_images(images, path, epoch=-1):
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

    for i, img in enumerate(images):
        inverse_normalize = transforms.Normalize(
            mean=[-1, -1, -1],
            std=[2, 2, 2]
        )
        # Inverse the normalization and transpose the shape
        # to [HEIGHT, WIDTH, CHANNELS]
        img = inverse_normalize(img).permute(1, 2, 0)
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
        - netD (nn.Module): The discriminator object to save.
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
        netG.state_dict(),
        os.path.join(loc, 'generator.pkl')
    )
    torch.save(
        netD.state_dict(),
        os.path.join(loc, 'discriminator.pkl')
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
