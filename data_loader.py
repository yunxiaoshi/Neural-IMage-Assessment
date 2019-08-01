# -*- coding: utf-8

import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data

class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name)
        annotations = self.annotations.iloc[idx, 1:].as_matrix()
        annotations = annotations.astype('float').reshape(-1, 1)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
