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
from model import STAGE1_G, STAGE1_D, STAGE2_G, STAGE2_D
from utils import KL_loss
from utils import compute_discriminator_loss, compute_generator_loss
from utils import save_images, save_checkpoints


class StackGAN(object):
    """StackGAN model class.

    Args:
        - dataset (string): Path to the [data].h5 file.
        - results (string): Output path for the results.
        - img_size (int, optional): Size for the images in the dataset.
        - transform (callable, optional): Optional transform to be applied
            on the image of a sample.
        - nc (int, optional): Number of channels for the images. (Default: 3)
        - text_dim (int, optional): Original text embeddings dimensions.
            (Default: 1024)
        - nt (int, optional): Projected embeddings dimensions. (Default: 128)
        - nz (int, optional): Dimension of the noise input. (Default: 100)
        - ngf (int, optional): Number of generator filters in the
            first convolutional layer. (Default: 128)
        - ndf (int, optional): Number of discriminator filters in the
            first convolutional layer. (Default: 64)
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
        img_size=64,
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
        """Initialize the StackGAN model."""
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
            self.image_transform = transforms.Compose([
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # Datasets
        self.train_ds = TxtDataset(
            data=dataset,
            split='train',
            img_size=img_size,
            transform=self.image_transform
        )
        self.test_ds = TxtDataset(
            data=dataset,
            split='test',
            img_size=img_size,
            transform=self.image_transform
        )

    def init_weights(self, m):
        """Initialize the weights.

        This method is applied to each layer of the Generator's and
        Discriminator's layers in order to initiliaze their weights
        and biases.
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def load_network_stageI(self, checkpoint=None):
        """Load the network for Stage I of StackGAN."""
        netG = STAGE1_G(self.ngf, self.nt, self.text_dim, self.nz)
        netG = nn.DataParallel(netG).to(self.device)

        netD = STAGE1_D(self.ndf, self.nt)
        netD = nn.DataParallel(netD).to(self.device)

        if checkpoint:
            try:
                gen_path = os.path.join(checkpoint, 'generator.pkl')
                gen_dict = torch.load(gen_path, map_location=lambda storage,
                                      loc: storage)
                netG.load_state_dict(gen_dict)

                dis_path = os.path.join(checkpoint, 'discriminator.pkl')
                dis_dict = torch.load(dis_path, map_location=lambda storage,
                                      loc: storage)
                netD.load_state_dict(dis_dict)
            except FileNotFoundError:
                print("[ERROR] Wrong checkpoint path or files don\'t exist.")
            except pickle.UnpicklingError as e:
                print("[ERROR] Something went wrong while loading"
                      " the state dictionary")
                print(f"[PICKLE] {e}")
                exit(1)
            except BaseException as err:
                print(f"[ERROR] {err}")
        else:
            netG.apply(self.init_weights)
            netD.apply(self.init_weights)

        return netG, netD

    def load_network_stageII(self, path, checkpoint=None):
        """Load the network for Stage II of StackGAN.

        Args:
            - path (string): Path to the state dict file of
                the Stage I generator.
        """
        Stage1_G = STAGE1_G(self.ngf, self.nt, self.text_dim, self.nz)
        netG = STAGE2_G(Stage1_G, self.ngf, self.nt, self.text_dim, self.nz)

        netD = STAGE2_D(self.ndf, self.nt)

        if checkpoint:
            netG.STAGE1_G = nn.DataParallel(netG.STAGE1_G).to(self.device)
            netG = nn.DataParallel(netG).to(self.device)
            netD = nn.DataParallel(netD).to(self.device)
            try:
                gen_path = os.path.join(checkpoint, 'generator.pkl')
                gen_dict = torch.load(gen_path, map_location=lambda storage,
                                      loc: storage)
                netG.load_state_dict(gen_dict)

                dis_path = os.path.join(checkpoint, 'discriminator.pkl')
                dis_dict = torch.load(dis_path, map_location=lambda storage,
                                      loc: storage)
                netD.load_state_dict(dis_dict)
            except FileNotFoundError:
                print("[ERROR] Wrong checkpoint path or files don\'t exist.")
                exit(1)
            except pickle.UnpicklingError as e:
                print("[ERROR] Something went wrong while loading"
                      " the state dictionary")
                print(f"[PICKLE] {e}")
                exit(1)
            except BaseException as err:
                print(f"[ERROR] {err}")
                exit(1)
        else:
            netG.apply(self.init_weights)
            netD.apply(self.init_weights)
            netG.STAGE1_G = nn.DataParallel(netG.STAGE1_G).to(self.device)
            try:
                state_dict = \
                    torch.load(path, map_location=lambda storage, loc: storage)
                netG.STAGE1_G.load_state_dict(state_dict)
            except FileNotFoundError:
                print("[ERROR] Wrong Stage I state dictionary path")
                exit(1)
            except pickle.UnpicklingError as e:
                print("[ERROR] Something went wrong while loading"
                      " the state dictionary")
                print(f"[PICKLE] {e}")
                exit(1)
            except BaseException as e:
                print("[ERROR] Stage I state dictionary file is corrupted")
                print(e)
                exit(1)

            netG = nn.DataParallel(netG).to(self.device)
            netD = nn.DataParallel(netD).to(self.device)

        return netG, netD

    def set_test(self, dataloader, batch_size=128, checkpoint=None):
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
                        example['embeddings'][:self.num_test % batch_size]

                    lines = example['texts'][:self.num_test % batch_size]
                    for line in lines:
                        f.write(f"[image-{n_line}]: {line}\n")
                        n_line += 1

                    break
                else:
                    real_images[i * batch_size:end] = example['images']
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
            save_images(real_images, self.images_dir, 0)

    def evaluate(self, netG, epoch):
        """Generate images for the num_test test samples selected.

        This method is called at the end of each training epoch. It
        forwards through the generator the test set's noise inputs and
        text embeddings, that were created during the call of the set_test
        method, and saved the generated images by calling the save_images
        function.

        Args:
            - netG (STAGE1_G or STAGE2_G): The generator object to evaluate.
            - epoch (int): The current epoch. Used for
                naming purposes.
        """
        netG.eval()
        with torch.no_grad():
            _, images, _, _ = netG(self.test_h, self.test_z)

        save_images(images, self.images_dir, epoch)

    def train(
        self,
        stage,
        stageI_path=None,
        num_epochs=600,
        batch_size=128,
        lr_G=0.0002,
        lr_D=0.0002,
        lr_decay=100,
        kl_coeff=2.0,
        adam_momentum=0.5,
        checkpoint_interval=10,
        checkpoint_path=None,
        num_workers=0,
    ):
        """Train the Generative Adversarial Network.

        This method implements the training of the StackGAN Network created
        for Text to Photo-realistic Image Synthesis with Stacked Generative
        Adversarial Networks. See here:
                https://arxiv.org/abs/1612.03242
        for further information about the training process.

        Args:
            - stage (int): StackGAN stage to train (1 or 2)
            - stageI_path (string, optional): path to the file of the saved
                state dictionary of the Stage I generator model.
                (Required for stage=2)
            - num_epochs (int, optional): Number of training epochs.
                (Default: 600)
            - batch_size (int, optional): Number of samples per batch.
                (Default: 128)
            - lr_G (float, optional): Learning rate for the generator's
                Adam optimizers. (Default: 0.0002)
            - lr_D (float, optional): Learning rate for the discriminator's
                Adam optimizers. (Default: 0.0002)
            - lr_decay (int, optional): Learning decay epoch step.
                (Default: 100)
            - kl_coeff (float, optional): Training coefficient for the
                Kullback-Leibler divergence. (Default: 2.0)
            - adam_momentum (float, optinal): Momentum value for the
                Adam optimizers' betas. (Default: 0.5)
            - checkpoint_interval (int, optional): Checkpoints will be saved
                every <checkpoint_interval> epochs. (Default: 10)
            - checkpoint_path (String, optional): Set this argument to continue
                training the models of the specified checkpoint. The path
                must have to following format:
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
        if stage not in [1, 2]:
            print("[ERROR] Stage must be either 1 or 2")
            exit(1)
        elif stage == 1:
            netG, netD = self.load_network_stageI(checkpoint_path)
        elif stageI_path is None and not checkpoint_path:
            print("[ERROR] Set path for the Stage I model")
            exit(1)
        else:
            netG, netD = \
                self.load_network_stageII(stageI_path, checkpoint_path)

        # Starting epoch
        starting_epoch = 1

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
        self.set_test(test_dataloader, batch_size, checkpoint_path)

        # Optimizers
        netG_params = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_params.append(p)

        if checkpoint_path:
            starting_epoch = \
                int(''.join(
                    c for c in checkpoint_path.split('-')[-1]
                    if c.isdigit()
                    )) + 1
            decays = starting_epoch // lr_decay
            lr_G = lr_G * (0.5 ** decays)
            lr_D = lr_D * (0.5 ** decays)

        optimizer_G = optim.Adam(
            netG_params,
            lr=lr_G,
            betas=(adam_momentum, 0.999)
        )
        optimizer_D = optim.Adam(
            netD.parameters(),
            lr=lr_D,
            betas=(adam_momentum, 0.999)
        )

        real = torch.FloatTensor(batch_size).fill_(1).to(self.device)
        fake = torch.FloatTensor(batch_size).fill_(0).to(self.device)

        generator_lr = lr_G
        discriminator_lr = lr_D

        training_start = datetime.now()
        print(
            f"\n{training_start.strftime('%d %B [%H:%M:%S] ')}"
            "Starting training..."
        )
        last_epoch = num_epochs
        for epoch in range(starting_epoch, num_epochs + 1):
            print(f"\nEpoch [{epoch}/{num_epochs}]")
            netG.train()
            netD.train()

            if epoch % lr_decay == 0 and epoch != starting_epoch:
                generator_lr *= 0.5
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, example in enumerate(train_dataloader):
                real_imgs = example['images'].to(self.device)
                txt_embeddings = example['embeddings'].to(self.device)

                noise = torch.randn(batch_size, self.nz).to(self.device)

                # Generate fake images
                _, fake_imgs, mu, logvar = \
                    netG(txt_embeddings, noise)
                fake_imgs = fake_imgs.to(self.device)

                # Update Discriminator
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                               real, fake,
                                               mu)
                errD.backward()
                optimizer_D.step()

                # Update Generator
                netG.zero_grad()
                errG = compute_generator_loss(netD, fake_imgs,
                                              real, mu)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * kl_coeff
                errG_total.backward()
                optimizer_G.step()

                # Append losses
                self.total_D_losses.append(errD.item())
                self.total_G_losses.append(errG.item())

                if (i % 20 == 0 or i == len(train_dataloader) - 1):
                    print(
                        f"Batch [{i + 1}/{len(train_dataloader)}]\tGenerator "
                        f"Loss: {errG.item()}\tDiscriminator "
                        f"Loss: {errD.item()}"
                    )
                else:
                    print(f"Batch [{i + 1}/{len(train_dataloader)}]", end="\r")

            self.evaluate(netG, epoch)
            if (epoch % checkpoint_interval == 0 or epoch == last_epoch):
                save_checkpoints(netG, netD, self.total_G_losses,
                                 self.total_D_losses, epoch,
                                 self.checkpoints_dir)

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
            keys = list(locals().keys())[:12]
            values = list(locals().values())[:12]
            for k, v in zip(keys, values):
                f.write(f"{k}: {v}\n\n")
            f.write(f"training duration: {duration}")
