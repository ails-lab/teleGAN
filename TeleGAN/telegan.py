"""This module implements the TeleGAN model for text to image synthesis task."""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import os
import random
import pickle
import numpy as np
from PIL import Image
from datetime import datetime

from dataset import TxtDataset
from loss import GeneratorLoss
from utils import KL_loss, copy_G_params
from utils import save_images, save_checkpoints
from model import Generator, DNET_BAW, DNET_COLORED, DNET_SR


class TeleGAN(object):
    """TeleGAN class for the Text To Image Synthesis task.

    Args:
        - dataset (string): Path to the [data].h5 file
        - results (string): Output path for the results
        - img_size (int, optional): Size for the images in the dataset
        - transform (callable, optional): Optional transform to be applied
            on the image of a sample (Default: Resize, RandomCrop and
            RandomHorizontalFlip)
        - nc (int, optional): Number of channels for the images (Default: 3)
        - text_dim (int, optional): Original text embeddings dimensions.
            (Default: 1024)
        - nt (int, optional): Projected embeddings dimensions (Default: 128)
        - nz (int, optional): Dimension of the noise input (Default: 100)
        - ngf (int, optional): Number of generator filters in the
            first convolutional layer (Default: 128)
        - ndf (int, optional): Number of discriminator filters in the
            first convolutional layer (Default: 64)
        - num_test (int, optional): Number of generated images for evaluation
            (Default: 50)
        - device (string, optional): Device to use for training
            ('cpu' or 'cuda') (Default: If there is a CUDA device
            available, it will be used for training)
    """

    def __init__(
        self,
        dataset,
        results,
        img_size=256,
        transform=None,
        nc=3,
        text_dim=1024,
        nt=128,
        nz=100,
        ngf=128,
        ndf=64,
        num_test=50,
        device=None
    ):
        """Initialize the TeleGAN model."""
        self.nz = nz
        self.nc = nc
        self.img_size = img_size
        self.num_test = num_test
        self.ngf = ngf
        self.text_dim = text_dim
        self.ndf = ndf
        self.nt = nt

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
            scale_size = int(self.img_size * 76 / 64)
            self.image_transform = transforms.Compose([
                transforms.Resize(
                    (scale_size, scale_size),
                    interpolation=Image.BICUBIC
                ),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip()
            ])

        # Datasets
        self.train_ds = TxtDataset(
            data=dataset,
            split='train',
            img_size=self.img_size,
            transform=self.image_transform
        )
        self.test_ds = TxtDataset(
            data=dataset,
            split='test',
            img_size=self.img_size,
            transform=self.image_transform
        )

    def init_weights(self, m):
        """Initialize the weights.

        This method is applied to each layer of the Generator's and
        Discriminator's layers in order to initiliaze their weights
        and biases.
        """
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or classname.find('Conv') != -1:
            nn.init.orthogonal_(m.weight.data, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            nn.init.orthogonal_(m.weight.data, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def load_networks(self, stage, path=None, checkpoint=None):
        """Load the generator and discriminator networks.

        This method initializes the Generator's and Discriminator's networks
        based on the number of branches.
        """
        netG = Generator(stage, self.ngf, self.nt, self.nz, self.text_dim)
        if stage == 1:
            netD = DNET_BAW(ndf=self.ndf, nef=self.nt)
        elif stage == 2:
            netD = DNET_COLORED(ndf=self.ndf, nef=self.nt)
        else:
            netD = DNET_SR(ndf=self.ndf)

        if checkpoint:
            try:
                gen_path = os.path.join(checkpoint, 'generator.pkl')
                gen_dict = torch.load(gen_path, map_location=lambda storage,
                                      loc: storage)
                netG.load_state_dict(gen_dict)

                dis_path = os.path.join(
                    checkpoint,
                    f'discriminator.pkl'
                )
                dis_dict = torch.load(
                    dis_path,
                    map_location=lambda storage,
                    loc: storage
                )
                netD.load_state_dict(dis_dict)
            except FileNotFoundError:
                print("[ERROR] Wrong checkpoint path or files don\'t exist.")
                exit(1)
            except pickle.UnpicklingError as e:
                print("[ERROR] Something went wrong while loading"
                      " the state dictionary")
                print(f"[PICKLE] {e}")
                exit(1)
            except BaseException as e:
                print(f"[ERROR] {e}")
                exit(1)
        else:
            netG.apply(self.init_weights)
            netD.apply(self.init_weights)

        if path:
            try:
                state_dict = \
                    torch.load(path, map_location=lambda storage, loc: storage)
                netG.load_state_dict(state_dict, strict=False)
            except FileNotFoundError:
                print(f"[ERROR] Wrong Stage {stage - 1} state dictionary path")
                exit(1)
            except pickle.UnpicklingError as e:
                print("[ERROR] Something went wrong while loading"
                      " the state dictionary")
                print(f"[PICKLE] {e}")
                exit(1)
            except BaseException as e:
                print(
                    f"[ERROR] Stage {stage - 1} state "
                    "dictionary file is corrupted"
                )
                print(e)
                exit(1)

        netG = nn.DataParallel(netG).to(self.device)
        netD = nn.DataParallel(netD).to(self.device)

        return netG, netD

    def set_test(self, stage, dataloader, batch_size=64, checkpoint=None):
        """Initialize the test set for evaluation.

        This method takes as input a dataloader to generate samples that will
        be used for the evaluation of the model. In order to check the
        performance of the model this test set must be fixed since the
        start of the training. It also calls the save_images method to save
        the corresponding real images of the test set inside the real-images
        directory.

        Args:
            - stage (int): The current stage of training
            - dataloader (Dataloader): The test dataloader
            - batch_size (int, optional): The batch size used while
                initializing the dataloader (Default: 64)
            - checkpoint (string, optional): Path to checkpoint's files
                (Default: None)
        """
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        if checkpoint:
            test_h_path = os.path.join(checkpoint, 'test_embds.pt')
            test_z_path = os.path.join(checkpoint, 'noise.pt')
            try:
                self.test_h = torch.load(
                    test_h_path, map_location=lambda storage, loc: storage
                ).to(self.device)
                self.test_z = torch.load(
                    test_z_path, map_location=lambda storage, loc: storage
                ).to(self.device)
            except FileNotFoundError:
                print("[ERROR] Wrong checkpoint path or files don\'t exist.")
                exit(1)
            return

        test_h_path = os.path.join(self.checkpoints_dir, 'test_embds.pt')
        test_z_path = os.path.join(self.checkpoints_dir, 'noise.pt')

        if stage == 3:
            image_size = 256
        else:
            image_size = 128

        if stage == 1:
            number_of_channels = 1
        else:
            number_of_channels = 3

        with open(f"{self.images_dir}/captions.txt", 'w') as f:
            real_images = torch.FloatTensor(
                self.num_test,
                number_of_channels,
                image_size,
                image_size
            ).to(self.device)
            n_line = 0
            for i, example in enumerate(dataloader):
                end = min((i + 1) * batch_size, self.num_test)
                if end == self.num_test:
                    real_images[i * batch_size:end] = \
                        example['images'][stage-1][:self.num_test % batch_size]
                    self.test_h[i * batch_size:end] = \
                        example['embeddings'][:self.num_test % batch_size]

                    lines = example['texts'][:self.num_test % batch_size]
                    for line in lines:
                        f.write(f"[image-{n_line}]: {line}\n")
                        n_line += 1

                    break
                else:
                    real_images[i * batch_size:end] = \
                        example['images'][stage-1]
                    self.test_h[i * batch_size:end] = example['embeddings']

                    lines = example['texts']
                    for line in lines:
                        f.write(f"[image-{n_line}]: {line}\n")
                        n_line += 1

            f.close()
            torch.save(self.test_h, test_h_path)
            torch.save(self.test_z, test_z_path)
            self.test_z.to(self.device)
            self.test_h.to(self.device)
            save_images(real_images, self.images_dir, stage, 0)

    def define_optimizers(self, G_lr, D_lr, adam_momentum=0.5):
        """Define the optimizers.

        This method initializes the Adam optimizers.

        Args:
            - G_lr (float): Learning rate for the Generator.
            - D_lr (float): Learning rate for the Discriminators.
            - adam_momentum (float, optional): Adam momentum. (Defualt: 0.5)
        """
        optimizerD = optim.Adam(self.netD.parameters(),
                                lr=D_lr,
                                betas=(adam_momentum, 0.999))

        optimizerG = optim.Adam(self.netG.parameters(),
                                lr=G_lr,
                                betas=(adam_momentum, 0.999))
        return optimizerG, optimizerD

    def evaluate(self, epoch, stage):
        """Generate images for the `num_test` test samples selected.

        This method is called at the end of each training epoch in order
        to evaluate the performance of the model during training by generating
        and saving images based on the test set's noise and text inputs.

        Args:
            - epoch (int): The current epoch of training
            - stage (int): The current stage of training
        """
        self.netG.eval()
        with torch.no_grad():
            images, _, _ = self.netG(self.test_z, self.test_h, train=False)

        save_images(images[-1], self.images_dir, stage, epoch)

    def train_Dnet(self, mu, real, fake, uncond_loss):
        """Train the Discriminator for stages 1 and 2.

        This method implements the training of the Discriminator for the
        Black and White and ENCOLOR stages.

        Args:
            - mu (float): Mu as calculated by the generator
            - real (tensor): Tensor containing '1' labels
            - fake (tensor): Tensor containing '0' labels
            - uncond_loss (float): Coefficient for the unconditional loss
        """
        batch_size = self.real_imgs.size(0)
        criterion, mu = self.criterion, mu

        netD, optD = self.netD, self.optimizerD
        real_imgs = self.real_imgs
        wrong_imgs = self.wrong_imgs
        fake_imgs = self.fake_imgs
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

    def train_Dnet_SR(self, real, fake):
        """Train the Discriminator for stage 3.

        This method implements the training of the Discriminator for
        the Super Resolution stage.

        Args:
            - real (tensor): Tensor containing '1' labels
            - fake (tensor): Tensor containing '0' labels
        """
        batch_size = self.real_imgs.size(0)

        netD, optD = self.netD, self.optimizerD
        real_imgs = self.real_imgs
        fake_imgs = self.fake_imgs
        #
        netD.zero_grad()
        # Forward
        real_labels = real[:batch_size]
        fake_labels = fake[:batch_size]
        # for real
        real_logits = netD(real_imgs).mean().to(self.device)
        fake_logits = netD(fake_imgs.detach()).mean().to(self.device)
        #
        errD = 1 - real_logits + fake_logits

        # backward
        errD.backward()
        # update parameters
        optD.step()
        return errD

    def train_Gnet(self, mu, logvar, real, uncond_loss):
        """Train the Generator for stages 1 and 2.

        This method implements the training of the Generator for
        the Black and White and ENCOLOR stages.

        Args:
            - mu (float): Mu as calculated by the generator
            - logvar (float): logvar as calculated by the generator
            - real (tensor): Tensor containing '1' labels
            - uncond_loss (float): Coefficient for the unconditional loss
        """
        self.netG.zero_grad()
        batch_size = self.real_imgs.size(0)
        criterion, mu, logvar = self.criterion, mu, logvar
        real_labels = real[:batch_size]
        outputs = self.netD(self.fake_imgs, mu)
        errG = criterion(outputs[0], real_labels)
        if len(outputs) > 1 and uncond_loss > 0:
            errG_patch = uncond_loss *\
                criterion(outputs[1], real_labels)
            errG = errG + errG_patch

        kl_loss = KL_loss(mu, logvar) * uncond_loss
        errG_total = errG + kl_loss
        errG_total.backward()
        self.optimizerG.step()
        return kl_loss, errG_total

    def train_Gnet_SR(self, real):
        """Train the Generator for stage 3.

        This method implements the training of the Generator for
        the Super Resolution stage.

        Args:
            - real (tensor): Tensor containing '1' labels
        """
        self.netG.zero_grad()
        batch_size = self.real_imgs.size(0)
        criterion = self.criterion
        fake_logits = self.netD(self.fake_imgs).mean().to(self.device)
        real_labels = real[:batch_size]

        errG = criterion(fake_logits, self.fake_imgs, self.real_imgs)
        errG.backward()
        self.optimizerG.step()

        return errG

    def train(
        self,
        stage,
        previous_stage_dict=None,
        num_epochs=600,
        batch_size=24,
        lr_G=0.0002,
        lr_D=0.0002,
        adam_momentum=0.5,
        uncond_loss=1.0,
        checkpoint_interval=10,
        checkpoint_path=None,
        num_workers=0
    ):
        """Train the Generative Adversarial Network.

        This method implements the training of TeleGAN. See here:
                http://artemis.cslab.ece.ntua.gr:8080/jspui/handle/123456789/17756
        for further information about the training process.

        Args:
            - stage (int): GAN's stage to train.
            - previous_stage_dict (string, optional): path to the file of
                the saved state dictionary of the previous stage.
                (Required for stage > 1)
            - num_epochs (int, optional): Number of training epochs.
                (Default: 600)
            - batch_size (int, optional): Number of samples per batch.
                (Default: 24)
            - lr_G (float, optional): Learning rate for the generator's
                Adam optimizers. (Default: 0.0002)
            - lr_D (float, optional): Learning rate for the discriminator's
                Adam optimizers. (Default: 0.0002)
            - adam_momentum (float, optinal): Momentum value for the
                Adam optimizers' betas. (Default: 0.5)
            - uncond_loss (float, optional): Coefficient for the unconditional
                loss. (Default: 1.0)
            - checkpoint_interval (int, optional): Checkpoints will be saved
                every `checkpoint_interval` epochs. (Default: 10)
            - checkpoint_path (String, optional): Set this argument to continue
                training the models of the specified checkpoint. The path
                must have the following format:
                    <'/path/to/results/[datetime]/checkpoints/epoch-[#]'>
                and inside there must be the following files:
                    - generator.pkl
                    - discriminator.pkl
                    - test_embds.pt
                    - noise.pt
                (Default: None)
            - num_workers (int, optional): Number of subprocesses to use
                for data loading. (Default: 0, whichs means that the data
                will be loaded in the main process.)
        """
        # For logging purposes
        saved_locals = locals()

        if stage not in [1, 2, 3]:
            print("[ERROR] Stage must be either 1, 2 or 3")
            exit(1)
        elif stage == 1:
            self.netG, self.netD = self.load_networks(
                stage, None, checkpoint_path)
        elif previous_stage_dict is None and not checkpoint_path:
            print("[ERROR] Set path for the previous stage model")
            exit(1)
        else:
            self.netG, self.netD = \
                self.load_networks(stage, previous_stage_dict, checkpoint_path)

        avg_param_G = copy_G_params(self.netG)

        # Starting epoch
        starting_epoch = 1
        if checkpoint_path:
            starting_epoch = \
                int(''.join(
                    c for c in checkpoint_path.split('-')[-1]
                    if c.isdigit()
                    )) + 1

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
        self.set_test(stage, test_dataloader, batch_size, checkpoint_path)

        # Optimizers
        self.optimizerG, self.optimizerD = \
            self.define_optimizers(lr_G, lr_D, adam_momentum)

        # Criterion
        if stage == 3:
            self.criterion = GeneratorLoss()
        else:
            self.criterion = nn.BCELoss()

        real = torch.FloatTensor(batch_size).fill_(0).to(self.device)
        fake = torch.FloatTensor(batch_size).fill_(1).to(self.device)

        training_start = datetime.now()
        print(
            f"\n{training_start.strftime('%d %B [%H:%M:%S] ')}"
            "Starting training..."
        )
        last_epoch = num_epochs
        for epoch in range(starting_epoch, num_epochs + 1):
            print(f"\nEpoch [{epoch}/{num_epochs}]")
            self.netG.train()
            self.netD.train()

            for i, example in enumerate(train_dataloader):
                self.real_imgs = example['images'][stage - 1].to(self.device)
                # Add decaying noise to real images for stability
                if stage != 3:
                    decay = 0.1 / (2 ** (epoch // 10))
                    inp_noise = ((decay ** 0.5) * torch.randn(
                        self.real_imgs.shape)).to(self.device)
                    self.real_imgs += inp_noise

                self.wrong_imgs = \
                    example['wrong_images'][stage-1].to(self.device)
                txt_embeddings = example['embeddings'].to(self.device)

                # Random noise
                if stage == 3:
                    noise = example['images'][1].to(self.device)
                else:
                    noise = torch.randn(batch_size, self.nz).to(self.device)

                # Generate fake images
                all_fake_imgs, mu, logvar = \
                    self.netG(noise, txt_embeddings)
                self.fake_imgs = all_fake_imgs[-1].to(self.device)
                if mu is not None:
                    mu = mu.to(self.device)

                # Flip labels every 100 batches (after 10th epoch)
                flip = i % 100 == 0 or i % 100 == 1
                if epoch > 10 and flip and stage != 3:
                    real, fake = fake, real

                # Update Discriminators
                if stage == 3:
                    errD_total = self.train_Dnet_SR(real, fake)
                else:
                    errD_total = self.train_Dnet(mu, real, fake,
                                                 uncond_loss)

                # Update Generator
                if stage == 3:
                    errG_total = self.train_Gnet_SR(real)
                else:
                    kl_loss, errG_total = \
                        self.train_Gnet(mu, logvar, real, uncond_loss)
                    for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                        avg_p.mul_(0.999).add_(0.001, p.data)

                # Append losses
                self.total_D_losses.append(float(errD_total.item()))
                self.total_G_losses.append(float(errG_total.item()))

                if (i % 20 == 0 or i == len(train_dataloader) - 1):
                    print(
                        f"Batch [{i + 1}/{len(train_dataloader)}]\tGenerator "
                        f"Loss: {errG_total.item()}\tDiscriminator "
                        f"Loss: {errD_total.item()}"
                    )
                else:
                    print(f"Batch [{i + 1}/{len(train_dataloader)}]", end="\r")

            self.evaluate(epoch, stage)
            if (epoch % checkpoint_interval == 0):
                save_checkpoints(self.netG, self.netD,
                                 self.total_G_losses, self.total_D_losses,
                                 epoch, self.checkpoints_dir)

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
            f.write("#################################\n")
            f.write("### INITIALIZATION PARAMETERS ###\n")
            f.write("#################################\n\n")
            for k, v in self.__dict__.items():
                if not isinstance(v, list):
                    f.write(f"{k}: {v}\n\n")
            f.write("#################################\n")
            f.write("###### TRAINING PARAMETERS ######\n")
            f.write("#################################\n\n")
            for k, v in saved_locals.items():
                f.write(f"{k}: {v}\n\n")
            f.write(f"training duration: {duration}")
