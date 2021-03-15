"""This module implements the dataset class for TeleGAN."""
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import io
import h5py
import numpy as np
from PIL import Image


class TxtDataset(Dataset):
    """TeleGAN dataset.

    Args:
        - data (string): Path to the h5 file with the data.
        - split (string): 'train', 'val' or 'test' split.
        - img_size (int, optional): Max size for the images in the dataset.
            (Default: 256)
        - transform (callable, optional): Optional transform to be applied
            on the image of a sample.
    """

    def __init__(self, data, split='train', img_size=256,
                 transform=None):
        """Initialize a TxtDataset.

        Other attributes:
        - hf (HDF5 group): The HDF5 group data of the dataset split.
        - keys (list): A list of string keys for the hf.
        """
        self.data = data
        self.split = split
        self.transform = transform

        self.img_sizes = [img_size // 2, img_size]

        self.hf = None
        self.keys = None
        with h5py.File(data, 'r') as f:
            self.ds_len = len(f[split])

    def __len__(self):
        """Return the length of the dataset split."""
        return self.ds_len

    def get_imgs(self, key):
        """Return the images corresponding to the key as a PIL Images."""
        img = self.hf[key]['image'][()]
        img = Image.open(io.BytesIO(img)).convert('RGB')

        bbox = None
        if "bounding_box" in self.hf[key].keys():
            bbox = self.hf[key]["bounding_box"][()]

        width, height = img.size
        # For the 'Birds' dataset we crop the image to the
        # corresponding bounding box in order for it to have
        # greater-than-0.75 object-image size ratio.
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])

        if self.transform is not None:
            img = self.transform(img)

        width, height = img.size
        half_size = transforms.Resize(
            (width // 2, height // 2),
            interpolation=Image.BICUBIC
        )

        BAW_norm = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        br_images = []
        br_images.append(BAW_norm(half_size(img)))
        br_images.append(normalize(half_size(img)))
        br_images.append(transforms.ToTensor()(img))

        return br_images

    def __getitem__(self, index):
        """Return a sample of the dataset."""
        if self.hf is None:
            self.hf = h5py.File(self.data, 'r')[self.split]
            self.keys = [str(key) for key in self.hf.keys()]

        key = self.keys[index]
        wrong_key = np.random.choice(self.keys)
        while wrong_key == key:
            wrong_key = np.random.choice(self.keys)

        img = self.get_imgs(key)
        wrong_img = self.get_imgs(wrong_key)

        embd_index = np.random.choice(len(self.hf[key]['embeddings']))
        embedding = self.hf[key]['embeddings'][()][embd_index]
        text = self.hf[key]['texts'][()][embd_index]

        example = {
            'keys': key,
            'images': img,
            'wrong_images': wrong_img,
            'embeddings': embedding,
            'texts': text
        }

        return example
