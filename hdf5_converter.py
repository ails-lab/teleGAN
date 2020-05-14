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
from glob import glob


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


def load_bounding_boxes(bounding_boxes_file, images_file):
    """Create a dictionary of bounding boxes points.

    This function is called for the 'Birds' dataset and
    creates a dictionary with the following structure:
        <image_name> : <bounding box>
    where bounding box is a list that contains the 4 points of the image
    that correspond to the bounding box of the bird image. The birds' images
    need to be cropped at that box in order to have greater-than-0.75
    object-image size ratios. (Used in the StackGAN implementation)
    """
    image_ids = {}
    bboxes = {}
    with open(images_file) as f:
        for l in f:
            name = l.split('/')[-1].split('.')[0]
            image_id = l.split(' ')[0]
            image_ids[image_id] = name
    with open(bounding_boxes_file) as f:
        for l in f:
            image_id = l.split(' ')[0]
            bbox = l.split(' ')[1:]
            bbox = [float(x) for x in bbox]
            bboxes[image_ids[image_id]] = bbox
    return bboxes


def convert(dataset, configs, n):
    """Create an HDF5 dataset.

    This function creates an .h5 file by collecting the corresponding
    input data from the different directories. The HDF Structure is as
    follows:

        - train / val / test
            - Example Name 1
                - name
                - class
                - image
                - bounding_box ('Birds' dataset only)
                - n embeddings
                - n texts

    Inputs:
        - dataset: the name of the dataset
        - configs: the configuration data of the config.json file
        - n: number of embeddings to keep (default: 10)
    Output:
        - Creates the <dataset>.h5 file in the same directory.
    """
    if dataset == 'birds':
        bboxes = load_bounding_boxes(
            configs["bboxes_file"],
            configs["images_file"]
        )
    else:
        bboxes = None

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
            tf = torchfile.load(
                embd_file.path,
                force_8bytes_long=True
            )

            img = open(
                os.path.join(
                    configs['images_path'],
                    tf[b'img'].decode('UTF-8')
                ),
                'rb'
            ).read()

            try:
                texts = np.array(
                    get_lines(
                        f"{configs['descriptions_path']}"
                        f"/{_class.name}/{name}.txt"
                    ),
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
            except FileNotFoundError:
                # Some classes in the CUB dataset have plural form in the
                # _icml directory but singular form in the text_c10 directory
                filename = (
                    f"{configs['descriptions_path']}/"
                    f"{_class.name[:3]}*/{name}.txt"
                )
                dest_list = glob(filename)
                if dest_list:
                    dest = dest_list[0]
                else:
                    print(f"No such file: {filename}")
                    sys.exit(1)

                texts = np.array(
                    get_lines(dest),
                    dtype=h5py.string_dtype(encoding='utf-8')
                )

            embds = tf[b'txt']

            if n < 10:
                rand_inds = np.random.choice(range(len(embds)), n)
                texts = texts[rand_inds]
                embds = embds[rand_inds]

            dt = h5py.special_dtype(vlen=str)

            example = split.create_group(name)
            example.create_dataset('name', data=name)
            example.create_dataset('class', data=_class.name)
            example.create_dataset('image', data=np.void(img))

            if bboxes:
                example.create_dataset('bounding_box', data=bboxes[name])

            example.create_dataset('embeddings', data=embds)
            example.create_dataset('texts', data=texts, dtype=dt)

        bar.next()
    bar.finish()
    hf.close()


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
    parser.add_argument(
        '-n',
        help="Number of embeddings [1,10]",
        default=10
    )

    # Parse the given arguments.
    args = parser.parse_args()
    dataset = args.dataset.lower()
    config = args.config
    n = int(args.n)

    if n < 1 or n > 10:
        print("n must be in [1,10]")

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
    convert(dataset, configs, n)
