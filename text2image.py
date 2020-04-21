import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from dataset import Txt2ImgDataset
from model import Generator, Discriminator

import os
import random
from datetime import datetime


# TODO:
# 1) INITIALIZE WEIGHTS
# 2) IMPLEMENT SAVE_IMGS METHOD
# 3) IMPLEMENT TEST METHOD
# 4) WRITE DOCSTRINGS


class Text2Image(object):
    def __init__(
        self,
        dataset,
        dataset_path,
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
        super(Train, self).__init__()
        self.nz = nz
        self.nc = nc
        self.img_size = img_size
        self.num_test = num_test
        self.total_G_losses = []
        self.total_D_losses = []
        self.test_z = torch.randn(num_test, nz, 1, 1)
        self.test_h = torch.FloatTensor(num_test, ne)

        self.image_transform = None
        if transform:
            self.image_transform = transform
        else:
            self.image_transform = transforms.Compose([
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1))
            ])

        # Datasets
        self.train_ds = Txt2ImgDataset(
            data=dataset_path,
            split='train',
            img_size=img_size,
            transform=self.image_transform
        )
        self.test_ds = Txt2ImgDataset(
            data=dataset_path,
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

    def set_test(self, dataloader, batch_size=64):
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
        self.save_images(real_images)

    def train(
        self,
        num_epochs=600,
        batch_size=64,
        lr=0.0002,
        int_beta=0.5,
        adam_momentum=0.5,
        num_workers=0
    ):
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

        real = torch.FloatTensor(batch_size, 1).fill_(1)
        fake = torch.FloatTensor(batch_size, 1).fill_(0)

        print("Starting Training...")
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

                if (i % 50 == 0):
                    print(
                        f"Batch [{i}/{len(train_dataloader)}]\tGenerator Loss:"
                        f" {G_loss}\tDiscriminator Loss: {D_loss}"
                    )

            # TODO: Save images and model
