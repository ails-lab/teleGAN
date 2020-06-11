import h5py
import io
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class TxtDataset(Dataset):
    """StackGANv2 dataset.

    Args:
        - data (string): Path to the h5 file with the data.
        - split (string): 'train', 'val' or 'test' split.
        - base_size (int, optional): Base size for the images in the dataset.
            (Default: 64)
        - transform (callable, optional): Optional transform to be applied
            on the image of a sample.
        - target_transform (callable, optional): Optional transform to be
            applied on the embeddings of a sample.
        - branch_num (int, optional): Number of branches in the model.
    """

    def __init__(self, data, split='train', base_size=64,
                 transform=None, target_transform=None, branch_num=3):
        """Initialize a TextDataset.

        Other attributes:
        - hf (HDF5 group): The HDF5 group data of the dataset split.
        - keys (list): A list of string keys for the hf.
        """
        self.data = data
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.branch_num = branch_num

        self.img_sizes = []
        for i in range(branch_num):
            self.img_sizes.append(base_size)
            base_size = base_size * 2

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

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        br_images = []
        for i in range(self.branch_num):
            if self.img_sizes[i] == img.size:
                br_images.append(normalize(img))
            else:
                scaled_image = transforms.Resize(
                    (self.img_sizes[i], self.img_sizes[i])
                )(img)
                br_images.append(normalize(scaled_image))

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

        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        example = {
            'keys': key,
            'images': img,
            'wrong_images': wrong_img,
            'embeddings': embedding,
            'texts': text
        }

        return example
