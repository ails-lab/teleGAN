import h5py
import json
import numpy as np
import os
import sys
import torchfile

from argparse import ArgumentParser
from matplotlib import image
from matplotlib import pyplot as plt
from progress.bar import Bar


def get_lines(file):
    """Create a list of lines.

    This function reads a file and appends each one of
    its lines to a list.

    Input:
        - file: path to the file
    Output:
        - lines: a list of the file's lines
    """
    lines = []
    with open(file) as f:
        for line in f:
            lines.append(line.strip())

    return lines


def convert(dataset, configs):
    """Create an HDF5 dataset.

    This function creates an .h5 file by collecting the corresponding
    input data from the different directories. The HDF Structure is as
    follows:

        - train / val / test
            - Example Name 1
                - name
                - class
                - image
                - embeddings
                - texts

    Inputs:
        - dataset: the name of the dataset
        - configs: the configuration data of the config.json file
    Output:
        - Creates the <dataset>.h5 file in the same directory.
    """
    hf = h5py.File(dataset + '.h5', 'w')

    train = hf.create_group('train')
    val = hf.create_group('val')
    test = hf.create_group('test')

    train_classes = get_lines(configs["train_file"])
    val_classes = get_lines(configs["val_file"])
    test_classes = get_lines(configs["test_file"])

    emb_path = configs["embeddings_path"]

    # Progress bar
    n_classes = sum(os.path.isdir(os.path.join(emb_path, i))
                    for i in os.listdir(emb_path))
    bar = Bar('Converting:', max=n_classes)

    for _class in os.scandir(emb_path):

        if not _class.is_dir():
            continue

        if _class.name in train_classes:
            split = train
        elif _class.name in val_classes:
            split = val
        elif _class.name in test_classes:
            split = test
        else:
            split = None

        if not split:
            continue

        for embd_file in os.scandir(_class.path):
            name = embd_file.name.split('.')[0]

            img = open(f"{configs['images_path']}/{name}.jpg", 'rb').read()

            texts = np.array(
                get_lines(
                    f"{configs['descriptions_path']}/{_class.name}/{name}.txt"
                ),
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            embds = torchfile.load(
                embd_file.path,
                force_8bytes_long=True
            )[b'txt']

            # The data contain 10 descriptions per image. As described
            # in the original paper, we will use only 5 of them.
            rand_inds = np.random.choice(range(len(embds)), 5)

            texts = texts[rand_inds]
            embds = embds[rand_inds]
            dt = h5py.special_dtype(vlen=str)

            for i, embd in enumerate(embds):
                example = split.create_group(name + '_' + str(i))
                example.create_dataset('name', data=name)
                example.create_dataset('class', data=_class.name)
                example.create_dataset('image', data=np.void(img))
                example.create_dataset('embeddings', data=embd)
                example.create_dataset('texts', data=texts[i], dtype=dt)
        bar.next()
    bar.finish()


if __name__ == "__main__":

    # Define the argument parser
    parser = ArgumentParser(description='Data to h5 converter.')
    required = parser.add_argument_group('required arguments')

    required.add_argument(
        '--dataset',
        help="Flowers or Birds",
        required=True
    )

    required.add_argument(
        '--config',
        help="Full path to the configuration file",
        required=True
    )

    # Parse the given arguments.
    args = parser.parse_args()
    dataset = args.dataset.lower()
    config = args.config

    # Check user inputs
    try:
        with open(config) as f:
            config_json = json.load(f)
            configs = config_json[dataset]
    except FileNotFoundError:
        print(f"No such file: '{config}'")
        sys.exit(1)
    except KeyError:
        print("Wrong dataset name. Use either 'Flowers' or 'Birds'.")
        sys.exit(1)

    # Create the <dataset>.h5 file
    convert(dataset, configs)
