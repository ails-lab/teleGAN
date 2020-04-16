import torch
import h5py
from torch.utils.data import Dataset


class Txt2ImgDataset(Dataset):
    """Text-to-Image synthesis dataset."""

    def __init__(self, data, split, img_size=64, transform=None):
        """Initialize a TextToImageDataset.

        Args:
           - data (string): Path to the h5 file with the data.
           - split (string): 'train', 'val' or 'test' split.
           - img_size (int, optional): Size for the images in the dataset.
           - transform (callable, optional): Optional transform to be applied
                on the image of a sample.
        Other attributes:
            - hf (HDF5 group): The HDF5 group data of the dataset split.
            - keys (list): A list of string keys for the hf.
        """
        self.data = data
        self.split = split
        self.img_size = img_size
        self.transform = transform

        self.hf = h5py.File(data, 'r')[split]
        self.keys = [str(key) for key in self.hf.keys()]

    def __len__(self):
        """Return the length of the dataset split."""
        return len(self.hf)

    def __getitem__(self, idx):
        """Return a sample of the dataset."""
        pass
