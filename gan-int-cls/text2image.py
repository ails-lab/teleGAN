import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import os
import random
import numpy as np
from PIL import Image
from datetime import datetime

from dataset import Txt2ImgDataset
from model import Generator, Discriminator
from utils import save_images, save_checkpoints


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
        num_test=200,
        device=None
    ):
        """Initialize the Text2Image model."""
        self.nz = nz
        self.nc = nc
        self.img_size = img_size
        self.num_test = num_test

        if not device:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print("[ERROR] CUDA is not available.")
                sys.exit(1)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            print("[ERROR] Wrong device input. ('cpu' or 'cuda')")
            sys.exit(1)

        self.total_G_losses = []
        self.total_D_losses = []
        self.test_z = torch.randn(num_test, nz, 1, 1)
        self.test_h = torch.FloatTensor(num_test, ne)
        self.results_dir = results
        self.images_dir = None
        self.checkpoints_dir = None

        self.image_transform = None
        if (transform and 'Compose' in torch.typename(transform)):
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

        self.discriminator = Discriminator(
            ne=ne,
            nt=nt,
            ndf=ndf
        )

        if (self.device.type == 'cuda'):
            self.generator = nn.DataParallel(self.generator).to(self.device)
            self.discriminator = \
                nn.DataParallel(self.discriminator).to(self.device)

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

    def init_weights(self, m):
        """Initialize the weights.

        This method is applied to each layer of the Generator's and
        Discriminator's layers in order to initialize their weights
        and biases.
        """
        name = torch.typename(m)
        if 'Conv' in name:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif 'BatchNorm' in name:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

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
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        with open(f"{self.images_dir}/captions.txt", 'w') as f:
            real_images = torch.FloatTensor(
                self.num_test,
                self.nc,
                self.img_size,
                self.img_size
            ).to(self.device)
            n_line = 0
            for i, example in enumerate(dataloader):
                end = min((i + 1) * batch_size, self.num_test)
                if end == self.num_test:
                    real_images[i * batch_size:end] = \
                        example['images'][:self.num_test % batch_size]
                    self.test_h[i * batch_size:end] = \
                        example['right_embds'][:self.num_test % batch_size]

                    lines = example['right_texts'][:self.num_test % batch_size]
                    for line in lines:
                        f.write(f"[image-{n_line}]: {line}\n")
                        n_line += 1

                    break
                else:
                    real_images[i * batch_size:end] = example['images']
                    self.test_h[i * batch_size:end] = example['right_embds']

                    lines = example['right_texts']
                    for line in lines:
                        f.write(f"[image-{n_line}]: {line}\n")
                        n_line += 1

            f.close()
            save_images(real_images, self.images_dir, 0)

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

        save_images(images, self.images_dir, epoch)

    def train(
        self,
        num_epochs=600,
        batch_size=64,
        lr=0.0002,
        int_beta=0.5,
        adam_momentum=0.5,
        checkpoint_interval=10,
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
            - checkpoint_interval (int, optional): Checkpoints will be saved
                every <checkpoint_interval> epochs. (Default: 10)
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

        real = torch.FloatTensor(batch_size).fill_(1).to(self.device)
        fake = torch.FloatTensor(batch_size).fill_(0).to(self.device)

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
                x = example['images'].to(self.device)
                h = example['right_embds'].to(self.device)
                h_hat = example['wrong_embds'].to(self.device)
                z = torch.randn(batch_size, self.nz, 1, 1).to(self.device)

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
                x_hat = self.generator(z, h).to(self.device)
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
                x_hat_emb = self.generator(z, h_emb).to(self.device)
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
            if (epoch % checkpoint_interval == 0):
                save_checkpoints(
                    self.generator,
                    self.discriminator,
                    self.total_G_losses,
                    self.total_D_losses,
                    epoch,
                    self.checkpoints_dir
                )

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
