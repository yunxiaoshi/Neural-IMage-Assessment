"""
file - dataset.py
Customized dataset class to loop through the AVA dataset and apply needed image augmentations for training.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms


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
        image = Image.open(img_name).convert('RGB')
        annotations = self.annotations.iloc[idx, 1:].to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


if __name__ == '__main__':

    # sanity check
    root = './data/images'
    csv_file = './data/train_labels.csv'
    train_transform = transforms.Compose([
        transforms.Scale(256), 
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dset = AVADataset(csv_file=csv_file, root_dir=root, transform=train_transform)
    train_loader = data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)
    for i, data in enumerate(train_loader):
        images = data['image']
        print(images.size())
        labels = data['annotations']
        print(labels.size())
