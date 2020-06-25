import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import os
import random
import numpy as np
import pickle

from PIL import Image
from datetime import datetime

from dataset import TxtDataset
from model import G_NET, D_NET64, D_NET128, D_NET256
from utils import save_images, save_checkpoints, copy_G_params
from utils import KL_loss, compute_mean_covariance


class StackGANv2(object):
    """StackGANv2 model class.

    Args:
        - dataset (string): Path to the [data].h5 file.
        - results (string): Output path for the results.
        - base_size (int, optional): Base size for the images in the dataset.
        - transform (callable, optional): Optional transform to be applied
            on the image of a sample.
        - nc (int, optional): Number of channels for the images. (Default: 3)
        - text_dim (int, optional): Original text embeddings dimensions.
            (Default: 1024)
        - nt (int, optional): Projected embeddings dimensions. (Default: 128)
        - nz (int, optional): Dimension of the noise input. (Default: 100)
        - ngf (int, optional): Number of generator filters in the
            first convolutional layer. (Default: 64)
        - ndf (int, optional): Number of discriminator filters in the
            first convolutional layer. (Default: 64)
        - branch_num (int, optional): Number of branches. (Default: 3)
        - num_test (int, optional): Number of generated images for evaluation
            (Default: 50)
        - device (string, optional): Device to use for training
            ('cpu' or 'cuda'). (Default: If there is a CUDA device
            available, it will be used for training. Otherwise CPU.)
    """

    def __init__(
        self,
        dataset,
        results,
        base_size=64,
        transform=None,
        nc=3,
        text_dim=1024,
        nt=128,
        nz=100,
        ngf=64,
        ndf=64,
        branch_num=3,
        num_test=50,
        device=None
    ):
        """Initialize the StackGANv2 model."""
        self.nz = nz
        self.nc = nc
        self.num_test = num_test
        self.ngf = ngf
        self.text_dim = text_dim
        self.ndf = ndf
        self.nt = nt
        self.branch_num = branch_num
        self.base_size = base_size
        self.max_size = base_size * (2 ** (branch_num - 1))

        if not device:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print("[ERROR] CUDA is not available")
                sys.exit(1)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            print("[ERROR] Wrong device input ('cpu' or 'cuda')")
            sys.exit(1)

        self.total_G_losses = []
        self.total_D_losses = []
        self.test_z = torch.randn(num_test, nz)
        self.test_h = torch.FloatTensor(num_test, text_dim)
        self.results_dir = results
        self.images_dir = None
        self.checkpoints_dir = None

        self.image_transform = None
        if (transform and 'Compose' in torch.typename(transform)):
            self.image_transform = transform
        else:
            if transform:
                print("[ERROR] Wrong transform input. Using default.")
            scale_size = int(self.max_size * 76 / 64)
            self.image_transform = transforms.Compose([
                transforms.Resize((scale_size, scale_size)),
                transforms.RandomCrop(self.max_size),
                transforms.RandomHorizontalFlip()
            ])

        # Datasets
        self.train_ds = TxtDataset(
            data=dataset,
            split='train',
            base_size=self.base_size,
            transform=self.image_transform,
            branch_num=self.branch_num
        )
        self.test_ds = TxtDataset(
            data=dataset,
            split='test',
            base_size=self.base_size,
            transform=self.image_transform,
            branch_num=self.branch_num
        )

    def init_weights(self, m):
        """Initialize the weights.

        This method is applied to each layer of the Generator's and
        Discriminator's layers in order to initiliaze their weights
        and biases.
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.orthogonal_(m.weight.data, 1.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            nn.init.orthogonal_(m.weight.data, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def load_networks(self):
        """Load the generator and discriminator networks.

        This method initializes the Generator's and Discriminator's networks
        based on the number of branches.
        """
        netG = G_NET(ngf=self.ngf, nef=self.nt, nz=self.nz,
                     text_dim=self.text_dim, branch_num=self.branch_num)
        netG.apply(self.init_weights)
        netG = nn.DataParallel(netG).to(self.device)

        netsD = []
        if self.branch_num > 0:
            netsD.append(D_NET64(ndf=self.ndf, nef=self.nt))
        if self.branch_num > 1:
            netsD.append(D_NET128(ndf=self.ndf, nef=self.nt))
        if self.branch_num > 2:
            netsD.append(D_NET256(ndf=self.ndf, nef=self.nt))

        for i in range(len(netsD)):
            netsD[i].apply(self.init_weights)
            netsD[i] = nn.DataParallel(netsD[i]).to(self.device)

        return netG, netsD

    def set_test(self, dataloader, batch_size):
        """Initialize the test set for evaluation.

        This method takes as input a dataloader to generate samples that will
        be used for the evaluation of the model. In order to check the
        performance of the model this test set must be fixed since the
        start of the training. It also calls the save_images method to save
        the corresponding real images of the test set in the real-images
        directory.

        Args:
            - dataloader (Dataloader): The test dataloader.
            - batch_size (int): The batch size used while
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
                self.max_size,
                self.max_size
            ).to(self.device)
            n_line = 0
            for i, example in enumerate(dataloader):
                end = min((i + 1) * batch_size, self.num_test)
                if end == self.num_test:
                    real_images[i * batch_size:end] = \
                        example['images'][-1][:self.num_test % batch_size]
                    self.test_h[i * batch_size:end] = \
                        example['embeddings'][:self.num_test % batch_size]

                    lines = example['texts'][:self.num_test % batch_size]
                    for line in lines:
                        f.write(f"[image-{n_line}]: {line}\n")
                        n_line += 1

                    break
                else:
                    real_images[i * batch_size:end] = example['images'][-1]
                    self.test_h[i * batch_size:end] = example['embeddings'][-1]

                    lines = example['texts']
                    for line in lines:
                        f.write(f"[image-{n_line}]: {line}\n")
                        n_line += 1

            f.close()
            save_images(real_images, self.images_dir, 0)

    def define_optimizers(self, G_lr, D_lr, adam_momentum=0.5):
        """Define the optimizers.

        This method initializes the optimizers based on the number of
        branches.

        Args:
            - G_lr (float): Learning rate for the Generator.
            - D_lr (float): Learning rate for the Discriminators.
            - adam_momentum (float, optional): Adam momentum. (Defualt: 0.5)
        """
        optimizersD = []
        for i in range(len(self.netsD)):
            opt = optim.Adam(self.netsD[i].parameters(),
                             lr=D_lr,
                             betas=(adam_momentum, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(self.netG.parameters(),
                                lr=G_lr,
                                betas=(adam_momentum, 0.999))
        return optimizerG, optimizersD

    def evaluate(self, epoch):
        """Generate images for the num_test test samples selected.

        This method is called at the end of each training epoch. It
        forwards through the generator the test set's noise inputs and
        text embeddings, that were created during the call of the set_test
        method, and saved the generated images by calling the save_images
        function.

        Args:
            - netG (STAGE1_G or STAGE2_G): The generator object to evaluate.
            - epoch (int): The current epoch. Used for naming purposes.
        """
        self.netG.eval()
        with torch.no_grad():
            images, _, _ = self.netG(self.test_z, self.test_h)

        save_images(images[-1], self.images_dir, epoch)

    def train_Dnet(self, idx, real_imgs, wrong_imgs, fake_imgs,
                   mu, real, fake, uncond_loss):
        """Train the Discriminators.

        This method implements the training of the <idx> Discriminator.

        Args:
            - idx (int): Index of the discriminator to be trained.
            - real_imgs (list): List of tensors containing the real
                images at the different branch sizes.
            - wrong_imgs (list): List of tensors containing wrong
                images at the different branch sizes.
            - fake_imgs (list): List of tensors containing generated
                images at the different branch sizes.
            - mu (float): Mu as calculated by the generator.
            - real (tensor): Tensor containing '1' labels.
            - fake (tensor): Tensor containing '0' labels.
            - uncond_loss (float): Coefficient for the unconditional loss.
        """
        batch_size = real_imgs[0].size(0)
        criterion, mu = self.criterion, mu

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = real_imgs[idx]
        wrong_imgs = wrong_imgs[idx]
        fake_imgs = fake_imgs[idx]
        #
        netD.zero_grad()
        # Forward
        real_labels = real[:batch_size]
        fake_labels = fake[:batch_size]
        # for real
        real_logits = netD(real_imgs, mu.detach())
        wrong_logits = netD(wrong_imgs, mu.detach())
        fake_logits = netD(fake_imgs.detach(), mu.detach())
        #
        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)
        if len(real_logits) > 1 and uncond_loss > 0:
            errD_real_uncond = uncond_loss * \
                criterion(real_logits[1], real_labels)
            errD_wrong_uncond = uncond_loss * \
                criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = uncond_loss * \
                criterion(fake_logits[1], fake_labels)
            #
            errD_real = errD_real + errD_real_uncond
            errD_wrong = errD_wrong + errD_wrong_uncond
            errD_fake = errD_fake + errD_fake_uncond
            #
            errD = errD_real + errD_wrong + errD_fake
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)
        # backward
        errD.backward()
        # update parameters
        optD.step()
        return errD

    def train_Gnet(self, real_imgs, fake_imgs,
                   mu, logvar, real, uncond_loss, color_coef=0.0):
        """Train the Generator.

        This method implements the training of the Generator.

        Args:
            - real_imgs (list): List of tensors containing the real
                images at the different branch sizes.
            - fake_imgs (list): List of tensors containing generated
                images at the different branch sizes.
            - mu (float): Mu as calculated by the generator.
            - logvar (float): logvar as calculated by the generator.
            - real (tensor): Tensor containing '1' labels.
            - uncond_loss (float): Coefficient for the unconditional loss.
            - color_coef (float, optional): Coefficient for the computation
                of the color consistency losses. (Default: 0.0)
        """
        self.netG.zero_grad()
        errG_total = 0
        batch_size = real_imgs[0].size(0)
        criterion, mu, logvar = self.criterion, mu, logvar
        real_labels = real[:batch_size]
        for i in range(len(self.netsD)):
            outputs = self.netsD[i](fake_imgs[i], mu)
            errG = criterion(outputs[0], real_labels)
            if len(outputs) > 1 and uncond_loss > 0:
                errG_patch = uncond_loss *\
                    criterion(outputs[1], real_labels)
                errG = errG + errG_patch
            errG_total = errG_total + errG

        # Compute color consistency losses
        if color_coef > 0:
            if len(self.netsD) > 1:
                mu1, covariance1 = compute_mean_covariance(fake_imgs[-1])
                mu2, covariance2 = \
                    compute_mean_covariance(fake_imgs[-2].detach())
                like_mu2 = color_coef * nn.MSELoss()(mu1, mu2)
                like_cov2 = color_coef * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
            if len(self.netsD) > 2:
                mu1, covariance1 = compute_mean_covariance(fake_imgs[-2])
                mu2, covariance2 = \
                    compute_mean_covariance(fake_imgs[-3].detach())
                like_mu1 = color_coef * nn.MSELoss()(mu1, mu2)
                like_cov1 = color_coef * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1

        kl_loss = KL_loss(mu, logvar) * uncond_loss
        errG_total = errG_total + kl_loss
        errG_total.backward()
        self.optimizerG.step()
        return kl_loss, errG_total

    def train(
        self,
        num_epochs=600,
        batch_size=24,
        lr_G=0.0002,
        lr_D=0.0002,
        adam_momentum=0.5,
        uncond_loss=0.1,
        color_coef=0.0,
        checkpoint_interval=10,
        num_workers=0
    ):
        """Train the Generative Adversarial Network.

        This method implements the training of the StackGANv2 Network created
        for Realistic Image Synthesis with Stacked Generative Adversarial
        Networks. See here:
                https://arxiv.org/abs/1710.10916
        for further information about the training process.

        Args:
            - num_epochs (int, optional): Number of training epochs.
                (Default: 600)
            - batch_size (int, optional): Number of samples per batch.
                (Default: 24)
            - lr_G (float, optional): Learning rate for the generator's
                Adam optimizers. (Default: 0.0002)
            - lr_D (float, optional): Learning rate for the discriminator's
                Adam optimizers. (Default: 0.0002)
            - adam_momentum (float, optional): Momentum value for the
                Adam optimizers' betas. (Default: 0.5)
            - uncond_loss (float, optional): Coefficient for the unconditional
                loss. (Default: 1.0)
            - color_coef (float, optional): Coefficient for the color
                consistency loss. (Default: 0.0 (not used))
            - checkpoint_interval (int, optional): Checkpoints will be saved
                every <checkpoint_interval> epochs. (Default: 10)
            - num_workers (int, optional): Number of subprocesses to use
                for data loading. (Default: 0, whichs means that the data
                will be loaded in the main process.)
        """
        self.netG, self.netsD = self.load_networks()
        avg_param_G = copy_G_params(self.netG)

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

        log_file = os.path.join(
            self.results_dir,
            date,
            'log.txt'
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
        self.optimizerG, self.optimizersD = \
            self.define_optimizers(lr_G, lr_D,
                                   adam_momentum)

        # Criterion
        self.criterion = nn.BCELoss()

        real = torch.FloatTensor(batch_size).fill_(1).to(self.device)
        fake = torch.FloatTensor(batch_size).fill_(0).to(self.device)

        training_start = datetime.now()
        print(
            f"\n{training_start.strftime('%d %B [%H:%M:%S] ')}"
            "Starting training..."
        )
        last_epoch = num_epochs - 1
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            self.netG.train()
            for netD in self.netsD:
                netD.train()

            for i, example in enumerate(train_dataloader):
                real_imgs = [ex.to(self.device) for ex in example['images']]
                wrong_imgs = [ex.to(self.device)
                              for ex in example['wrong_images']]
                txt_embeddings = example['embeddings'].to(self.device)

                noise = torch.randn(batch_size, self.nz).to(self.device)

                # Generate fake images
                fake_imgs, mu, logvar = \
                    self.netG(noise, txt_embeddings)
                fake_imgs = [f_i.to(self.device) for f_i in fake_imgs]

                # Update Discriminators
                errD_total = 0
                for j in range(len(self.netsD)):
                    errD = self.train_Dnet(j, real_imgs, wrong_imgs,
                                           fake_imgs, mu, real, fake,
                                           uncond_loss)
                    errD_total += errD

                # Update Generator
                kl_loss, errG_total = \
                    self.train_Gnet(real_imgs, fake_imgs,
                                    mu, logvar, real, uncond_loss,
                                    color_coef=color_coef)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # Append losses
                self.total_D_losses.append(errD_total.data)
                self.total_G_losses.append(errG_total.data)

                if (i % 20 == 0 or i == len(train_dataloader) - 1):
                    print(
                        f"Batch [{i + 1}/{len(train_dataloader)}]\tGenerator "
                        f"Loss: {errG_total.data}\tDiscriminator "
                        f"Loss: {errD_total.data}"
                    )
                else:
                    print(f"Batch [{i + 1}/{len(train_dataloader)}]", end="\r")

            self.evaluate(epoch + 1)
            if (epoch % checkpoint_interval == 0 or epoch == last_epoch):
                save_checkpoints(self.netG, self.netsD,
                                 self.total_G_losses, self.total_D_losses,
                                 epoch + 1, self.checkpoints_dir)

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
        with open(log_file, 'w') as f:
            f.write("### INITIALIZATION PARAMETERS ###\n\n")
            for k, v in self.__dict__.items():
                f.write(f"{k}: {v}\n\n")
            f.write("### TRAINING PARAMETERS ###\n\n")
            keys = list(locals().keys())[:11]
            values = list(locals().values())[:11]
            for k, v in zip(keys, values):
                if k == "avg_param_G":
                    continue
                f.write(f"{k}: {v}\n\n")
            f.write(f"training duration: {duration}")
