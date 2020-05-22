"""Pascal VOC 2012 Semantic Segmentation Dataset.
The PASCAL Visual Object Classes
    http://host.robots.ox.ac.uk/pascal/VOC/
Code partially borrowed from:
    https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/pascal_voc/segmentation.py.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as standard_transforms

from config import Config

config = Config()

randomCrop = standard_transforms.RandomCrop(config.input_size)
centerCrop = standard_transforms.CenterCrop(config.input_size)
toTensor = standard_transforms.ToTensor()
toPIL = standard_transforms.ToPILImage()


class PascalVOC(data.Dataset):
    NUM_CLASS = 21
    CLASSES = ("background", "airplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorcycle", "person", "potted-plant", "sheep", "sofa", "train",
               "tv")

    def __init__(self, root='/home/ubuntu/workspace/us-seg-lib/data/pascal/VOC2012/',
                 input_transforms=None, split='train', mode='train'):
        self.root = root
        self.split = split
        self.mode = mode

        self.img_paths, self.mask_paths = _get_pascal_pairs(self.root, self.split)

        self.transforms = input_transforms

    def __getitem__(self, index):
        img_path, mask_path = self.img_paths[index], self.mask_paths[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        input = img
        output = None
        if self.mode == 'train':
            input = self.transforms(input)
            input = toPIL(input)
            output = input.copy()
            if self.mode == "train" and config.variationalTranslation > 0:
                output = randomCrop(input)
            input = toTensor(centerCrop(input))
            output = toTensor(output)

        elif self.mode == 'val':
            input = toTensor(img)
            mask = toTensor(mask)

        return input, output, mask

    def __len__(self):
        return len(self.img_paths)


def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    target[target == 255] = -1
    return target


def _get_pascal_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder, split_file, split):
        """get image and mask path pair"""
        img_paths = []
        mask_paths = []

        with open(os.path.join(split_file), "r") as lines:
            for line in lines:
                _image = os.path.join(img_folder, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                img_paths.append(_image)
                if split != 'test':
                    _mask = os.path.join(mask_folder, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    mask_paths.append(_mask)

        if split != 'test':
            assert (len(img_paths) == len(mask_paths))

        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    mask_folder = os.path.join(folder, 'SegmentationClass')
    img_folder = os.path.join(folder, 'JPEGImages')

    splits_folder = os.path.join(folder, 'ImageSets/Segmentation')

    if split == 'train':
        split_file = os.path.join(splits_folder, 'trainval.txt')
    elif split == 'val':
        split_file = os.path.join(splits_folder, 'val.txt')
    elif split == 'test':
        split_file = os.path.join(splits_folder, 'test.txt')
    else:
        raise RuntimeError('Unknown dataset split.')

    img_paths, mask_paths = get_path_pairs(img_folder, mask_folder, split_file, split)

    return img_paths, mask_paths
