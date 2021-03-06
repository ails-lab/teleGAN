import torch
import h5py
import io
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class Txt2ImgDataset(Dataset):
    """Text-to-Image synthesis dataset.

    Args:
        - data (string): Path to the h5 file with the data.
        - split (string): 'train', 'val' or 'test' split.
        - img_size (int, optional): Size for the images in the dataset.
            (Default: 64)
        - transform (callable, optional): Optional transform to be applied
            on the image of a sample.
    """

    def __init__(self, data, split, img_size=64, transform=None):
        """Initialize a TextToImageDataset.

        Other attributes:
        - hf (HDF5 group): The HDF5 group data of the dataset split.
        - keys (list): A list of string keys for the hf.
        """
        self.data = data
        self.split = split
        self.img_size = img_size
        self.transform = transform

        self.hf = None
        self.keys = None
        with h5py.File(data, 'r') as f:
            self.ds_len = len(f[split])

    def __len__(self):
        """Return the length of the dataset split."""
        return self.ds_len

    def get_img(self, key):
        """Return the image corresponding to the key as a PIL Image."""
        img = self.hf[key]['image'][()]

        img = Image.open(io.BytesIO(img)).convert('RGB')

        if (self.transform) is not None:
            # Transform contains RandomCrop((img_size,img_size))
            img = img.resize((self.img_size + 20, self.img_size + 20))
            img = self.transform(img)
        else:
            img = img.resize((self.img_size, self.img_size))

        return np.asarray(img)

    def __getitem__(self, idx):
        """Return a sample of the dataset."""
        if self.hf is None:
            self.hf = h5py.File(self.data, 'r')[self.split]
            self.keys = [str(key) for key in self.hf.keys()]

        key = self.keys[idx]

        img = self.get_img(key)

        embd_index = np.random.choice(len(self.hf[key]['embeddings']))
        right_embd = self.hf[key]['embeddings'][()][embd_index]
        right_txt = self.hf[key]['texts'][()][embd_index]

        key_wrong = np.random.choice(self.keys)
        while key_wrong == key:
            key_wrong = np.random.choice(self.keys)
        embd_index_wrong = np.random.choice(
            len(self.hf[key_wrong]['embeddings'])
        )
        wrong_embd = self.hf[key_wrong]['embeddings'][()][embd_index_wrong]
        wrong_txt = self.hf[key_wrong]['texts'][()][embd_index_wrong]

        example = {
            'keys': key,
            'images': img,
            'right_embds': right_embd,
            'right_texts': right_txt,
            'wrong_embds': wrong_embd,
            'wrong_texts': wrong_txt
        }

        return example
