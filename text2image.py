import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import os
import random
import numpy as np
from PIL import Image
from datetime import datetime
from matplotlib import pyplot as plt

from dataset import Txt2ImgDataset
from model import Generator, Discriminator

# TODO: ADD CUDA SUPPORT


class Text2Image(object):
    """Text2Image class for the Text To Image Synthesis GAN.

    Args:
        - dataset (string): Path to the [data].h5 file.
        - results (string): Output path for the results.
        - img_size (int, optional): Size for the images in the dataset.
        - transform (callable, optional): Optional transform to be applied
            on the image of a sample.
        - nc (int, optional): Number of channels for the images. (Default: 3)
        - ne (int, optional): Original embeddings dimensions. (Default: 1024)
        - nt (int, optional): Projected embeddings dimensions. (Default: 128)
        - nz (int, optional): Dimension of the noise input. (Default: 100)
        - ngf (int, optional): Number of generator filters in the
            first convolutional layer. (Default: 128)
        - ndf (int, optional): Number of discriminator filters in the
            first convolutional layer. (Default: 64)
        - num_test (int, optional): Number of generated images for evaluation
            (Default: 200)
    """

    def __init__(
        self,
        dataset,
        results,
        img_size=64,
        transform=None,
        nc=3,
        ne=1024,
        nt=128,
        nz=100,
        ngf=128,
        ndf=64,
        num_test=200
    ):
        """Initialize the Text2Image model."""
        self.nz = nz
        self.nc = nc
        self.img_size = img_size
        self.num_test = num_test
        self.total_G_losses = []
        self.total_D_losses = []
        self.test_z = torch.randn(num_test, nz, 1, 1)
        self.test_h = torch.FloatTensor(num_test, ne)
        self.results_dir = results
        self.images_dir = None
        self.checkpoints_dir = None

        self.image_transform = None
        if transform:
            self.image_transform = transform
        else:
            self.image_transform = transforms.Compose([
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Datasets
        self.train_ds = Txt2ImgDataset(
            data=dataset,
            split='train',
            img_size=img_size,
            transform=self.image_transform
        )
        self.test_ds = Txt2ImgDataset(
            data=dataset,
            split='test',
            img_size=img_size,
            transform=self.image_transform
        )

        # Networks
        self.generator = Generator(
            ne=ne,
            nt=nt,
            nz=nz,
            ngf=ngf
        )
        self.generator.apply(self.init_weights)

        self.discriminator = Discriminator(
            ne=ne,
            nt=nt,
            ndf=ndf
        )
        self.discriminator.apply(self.init_weights)

    def save_images(self, images, epoch=-1):
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
            - epoch (int, optional): The current epoch. It is used for
                naming purposes. If not given as input, the images will
                be saved inside the 'other_images' directory.
        """
        if epoch == -1:
            loc = os.path.join(
                self.images_dir,
                'other-images'
            )
        elif epoch == 0:
            loc = os.path.join(
                self.images_dir,
                'real-images'
            )
        else:
            loc = os.path.join(
                self.images_dir,
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
            img = (img.numpy() * 255).astype(np.uint8)

            im = Image.fromarray(img)
            im.save(os.path.join(loc, f'image-{i}.png'))

    def save_checkpoints(self, epoch):
        """Save Generator and Discriminator states along with a plot.

        This method saves the Generator and Discriminator states and a plot
        of their total losses (until the current batch) to the following files:
            - generator.pkl
            - discriminator.pkl
            - losses.png
        inside of the [path to checkpoints directory]/epoch-[#]/ directory.

        Args:
            - epoch (int): The current epoch. It is used for
                naming purposes.
        """
        loc = os.path.join(
                self.checkpoints_dir,
                f'epoch-{epoch}'
        )
        if not os.path.exists(loc):
            os.makedirs(loc)

        torch.save(
            self.generator.state_dict(),
            os.path.join(loc, 'generator.pkl')
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(loc, 'discriminator.pkl')
        )

        plt.figure()
        plt.grid()
        x = np.arange(len(self.total_G_losses))
        y_G = self.total_G_losses
        y_D = self.total_D_losses
        plt.plot(x, y_G, 'b', label='Generator losses')
        plt.plot(x, y_D, 'r', label='Discriminator losses')
        plt.legend(loc="upper right")
        plt.title('Generator and Discriminator losses')
        plt.xlabel('Number of training batches')
        plt.ylabel('Loss')
        plt.savefig(
            os.path.join(loc, 'losses.png')
        )

    def set_test(self, dataloader, batch_size=64):
        """Initialize the test set for evaluation.

        This method takes as input a dataloader to generate samples that will
        be used for the evaluation of the model. In order to check the
        performance of the model this test set must be fixed since the
        start of the training. It also calls the save_images method to save
        the corresponding real images of the test set in the real-images
        directory.

        Args:
            - dataloader (Dataloader): The test dataloader.
            - batch_size (int, optional): The batch size used while
                initializing the dataloader.
        """
        real_images = torch.FloatTensor(
            self.num_test,
            self.nc,
            self.img_size,
            self.img_size
        )
        for i, example in enumerate(dataloader):
            end = min((i + 1) * batch_size, self.num_test)
            if end == self.num_test:
                real_images[i * batch_size:end] = \
                    example['images'][:self.num_test % batch_size]
                self.test_h[i * batch_size:end] = \
                    example['right_embds'][:self.num_test % batch_size]
                break
            else:
                real_images[i * batch_size:end] = example['images']
                self.test_h[i * batch_size:end] = example['right_embds']

        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.save_images(real_images, 0)

    def init_weights(self, m):
        """Initialize the weights.

        This method is applied to each layer of the Generator's and
        Discriminator's layers in order to initliaze theirs weights
        and biases.
        """
        name = torch.typename(m)
        if 'Conv' in name:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif 'BatchNorm' in name:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def evaluate(self, epoch):
        """Generate images for the num_test test samples selected.

        This method is called at the end of each training epoch. It
        forwards through the generator the test set's noise inputs and
        text embeddings, that were created during the call of the set_test
        method, and saved the generated images by calling the save_images
        method.

        Args:
            - epoch (int): The current epoch. It is used for
                naming purposes.
        """
        self.generator.eval()
        with torch.no_grad():
            images = self.generator(self.test_z, self.test_h)

        self.save_images(images, epoch)

    def train(
        self,
        num_epochs=600,
        batch_size=64,
        lr=0.0002,
        int_beta=0.5,
        adam_momentum=0.5,
        num_workers=0
    ):
        """Train the Generative Adversarial Network.

        This method implements the training of the Generative Adversarial
        Network created for the Text to Image Synthesis. See here:
                https://arxiv.org/abs/1605.05396
        for further information about the training process.

        Args:
            - num_epochs (int, optional): Number of training epochs.
                (Default: 600)
            - batch_size (int, optional): Number of samples per batch.
                (Default: 64)
            - lr (float, optional): Learning rate for the Adam optimizers.
                (Default: 0.0002)
            - int_beta (float, optional): Beta value for the manifold
                interpolation of the text embeddings. (Default: 0.5)
            - adam_momentum (float, optinal): Momentum value for the
                Adam optimizers' betas. (Default: 0.5)
            - num_workers (int, optional): Number of subprocesses to use
                for data loading. (Default: 0, whichs means that the data
                will be loaded in the main process.)
        """
        # Results directory
        date = datetime.now().strftime("%d-%b-%Y (%H.%M)")
        self.images_dir = os.path.join(
            self.results_dir,
            date,
            'images'
        )
        self.checkpoints_dir = os.path.join(
            self.results_dir,
            date,
            'checkpoints'
        )

        # Dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers
        )
        test_dataloader = torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers
        )

        # Set test dataset
        self.set_test(test_dataloader, batch_size)

        # Optimizers
        optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(adam_momentum, 0.999)
        )
        optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(adam_momentum, 0.999)
        )

        # Criterion
        criterion = nn.BCELoss()

        real = torch.FloatTensor(batch_size).fill_(1)
        fake = torch.FloatTensor(batch_size).fill_(0)

        training_start = datetime.now()
        print(
            f"\n{training_start.strftime('%d %B [%H:%M:%S] ')}"
            "Starting training..."
        )
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch [{epoch}/{num_epochs}]")
            self.generator.train()
            self.discriminator.train()

            for i, example in enumerate(train_dataloader):
                x = example['images']
                h = example['right_embds']
                h_hat = example['wrong_embds']
                z = torch.randn(batch_size, self.nz, 1, 1)

                # Update Discriminator
                optimizer_D.zero_grad()
                D_loss = 0

                # Real Image - Right Text
                D_real_image = self.discriminator(x, h).view(-1)
                D_real_image_loss = criterion(D_real_image, real)
                D_real_image_loss.backward()
                D_loss += D_real_image_loss.item()

                # Real Image - Wrong Text
                D_wrong_text = self.discriminator(x, h_hat).view(-1)
                D_wrong_text_loss = criterion(D_wrong_text, fake)*0.5
                D_wrong_text_loss.backward()
                D_loss += D_wrong_text_loss.item()

                # Fake Image - Right Text
                x_hat = self.generator(z, h)
                D_fake_image = self.discriminator(x_hat.detach(), h).view(-1)
                D_fake_image_loss = criterion(D_fake_image, fake) * 0.5
                D_fake_image_loss.backward()
                D_loss += D_fake_image_loss.item()

                # Update
                optimizer_D.step()

                # Update Generator
                optimizer_G.zero_grad()
                G_loss = 0

                # Image loss
                D_fake_image = self.discriminator(x_hat, h).view(-1)
                G_fake_image_loss = criterion(D_fake_image, real)
                G_fake_image_loss.backward()
                G_loss += G_fake_image_loss.item()

                # Embeddings loss
                h_emb = int_beta * h + (1 - int_beta) * h_hat
                x_hat_emb = self.generator(z, h_emb)
                D_emb = self.discriminator(x_hat_emb, h_emb).view(-1)
                G_emb_loss = criterion(D_emb, fake)
                G_emb_loss.backward()
                G_loss += G_emb_loss.item()

                # Update
                optimizer_G.step()

                # Append losses
                self.total_D_losses.append(D_loss)
                self.total_G_losses.append(G_loss)

                if (i % 20 == 0 or i == len(train_dataloader) - 1):
                    print(
                        f"Batch [{i + 1}/{len(train_dataloader)}]\tGenerator "
                        f"Loss: {G_loss}\tDiscriminator Loss: {D_loss}"
                    )
                else:
                    print(f"Batch [{i + 1}/{len(train_dataloader)}]", end="\r")

            self.evaluate(epoch)
            self.save_checkpoints(epoch)

        training_end = datetime.now()
        print(
            f"\n{training_end.strftime('%d-%b [%H:%M:%S] ')}"
            "Finished training."
        )
        duration = (training_end - training_start)
        print(
            "Training duration: "
            f"{duration.days} days, {duration.seconds // 3600} hours"
            f" and {(duration.seconds // 60) % 60} minutes"
        )
