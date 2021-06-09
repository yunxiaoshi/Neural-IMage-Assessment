"""
file - test.py
Simple quick script to evaluate model on test images.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model.model import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to pretrained model')
parser.add_argument('--test_csv', type=str, help='test csv file')
parser.add_argument('--test_images', type=str, help='path to folder containing images')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--predictions', type=str, help='output file to store predictions')
args = parser.parse_args()

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)

try:
    model.load_state_dict(torch.load(args.model))
    print('successfully loaded model')
except:
    raise

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

model.eval()

test_transform = transforms.Compose([
    transforms.Scale(256), 
    transforms.RandomCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

test_df = pd.read_csv(args.test_csv, header=None)
test_imgs = test_df[0]
pbar = tqdm(total=len(test_imgs))

mean, std = 0.0, 0.0
for i, img in enumerate(test_imgs):
    im = Image.open(os.path.join(args.test_images, str(img) + '.jpg'))
    im = im.convert('RGB')
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)
    for j, e in enumerate(out, 1):
        mean += j * e
    for k, e in enumerate(out, 1):
        std += e * (k - mean) ** 2
    std = std ** 0.5
    gt = test_df[test_df[0] == img].to_numpy()[:, 1:].reshape(10, 1)
    gt_mean = 0.0
    for l, e in enumerate(gt, 1):
        gt_mean += l * e
    # print(str(img) + ' mean: %.3f | std: %.3f | GT: %.3f' % (mean, std, gt_mean))
    if not os.path.exists(args.predictions):
        os.makedirs(args.predictions)
    with open(os.path.join(args.predictions, 'pred.txt'), 'a') as f:
        f.write(str(img) + ' mean: %.3f | std: %.3f | GT: %.3f\n' % (mean, std, gt_mean))
    mean, std = 0.0, 0.0
    pbar.update()
