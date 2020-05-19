"""BSD500 Image Segmentation Dataset.
Contour Detection and Hierarchical Image Segmentation
    https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy.io import loadmat

import torch
from torch.utils import data
import torchvision.transforms as standard_transforms

from ..config import Config

config = Config()

randomCrop = standard_transforms.RandomCrop(config.input_size)
centerCrop = standard_transforms.CenterCrop(config.input_size)
toTensor = standard_transforms.ToTensor()
toPIL = standard_transforms.ToPILImage()


class BSDS500(data.Dataset):
    def __init__(self, root='/home/ubuntu/workspace/us-seg-lib/data/BSR/BSDS500/data',
                 split='train', mode='train', base_size=224, transforms=None):
        self.root = root
        self.split = split
        self.mode = mode

        self.img_paths, self.mask_paths = _get_bsd_pairs(self.root, self.split)

        self.base_size = base_size

        # Geometric image transformations
        self.transforms = None
        if transforms is not None:
            self.transforms = transforms

        # mask transformations
        self.target_transform = transforms.Compose([
            transforms.Resize(config.input_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path, mask_path = self.img_paths[index], self.mask_paths[index]
        raw_img = Image.open(img_path).convert('RGB')

        raw_masks, mean_mask = _loadmask(mask_path)
        gt_masks = []
        mean_mask = Image.fromarray(mean_mask.astype(np.float))

        input = self.transforms(raw_img)

        input = standard_transforms.ToPILImage(input)
        output = input.copy()

        if self.mode == "train" and config.variationalTranslation > 0:
            output = randomCrop(input)
        input = toTensor(centerCrop(input))
        output = toTensor(output)

        mask = self.target_transform(mean_mask)

        return input, output

    def __len__(self):
        return len(self.img_paths)


def _get_bsd_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        """get image and mask path pair"""
        img_paths = []
        mask_paths = []

        for filename in os.listdir(img_folder):
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)

                maskname = filename.replace('.jpg', '.mat')
                maskpath = os.path.join(mask_folder, maskname)

                if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask or image:', imgpath, maskpath)

        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val', 'test'):
        img_folder = os.path.join(folder, 'images/' + split)
        mask_folder = os.path.join(folder, 'groundTruth/' + split)

        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)

        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')

        train_img_folder = os.path.join(folder, 'images/train')
        train_mask_folder = os.path.join(folder, 'groundTruth//train')

        val_img_folder = os.path.join(folder, 'images/val')
        val_mask_folder = os.path.join(folder, 'groundTruth//val')

        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)

        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths

    return img_paths, mask_paths


def _loadmask(mask_path):
    raw_masks = []
    mean_mask = None
    mask_mat = loadmat(mask_path)['groundTruth']

    idx = mask_mat.shape[1]
    for i in range(idx):
        seg = mask_mat[0, i][0, 0][0]

        if i == 0:
            mean_mask = seg
        else:
            mean_mask = mean_mask + seg
        raw_masks.append(seg)

    mean_mask = mean_mask / idx
    raw_masks = np.array(raw_masks)

    return raw_masks, mean_mask.astype(np.int)
