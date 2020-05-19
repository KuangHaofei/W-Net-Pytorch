# Converts Berkeley segmentation dataset segmentation format files to .npy arrays
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

train_destination = "/home/ubuntu/workspace/W-Net-Pytorch/datasets/BSDS300/train/segmentations"
test_destination = "/home/ubuntu/workspace/W-Net-Pytorch/datasets/BSDS300/test/segmentations"

iid_train = np.loadtxt("/home/ubuntu/workspace/W-Net-Pytorch/datasets/BSDS300/iids_train.txt").astype(np.int)
iid_test = np.loadtxt("/home/ubuntu/workspace/W-Net-Pytorch/datasets/BSDS300/iids_test.txt").astype(np.int)

iid_train = iid_train.tolist()
iid_test = iid_test.tolist()

def convertAndSave(filepath, filename):
    f = open(filepath, 'r')
    w, h = (0, 0)
    for line in f:
        if 'width' in line:
            w = int(line.split(' ')[1])
        if 'height' in line:
            h = int(line.split(' ')[1])
        if 'data' in line:
            break

    seg = np.zeros((h, w))
    for line in f:
        s, r, c1, c2 = map(lambda x: int(x), line.split(' '))
        seg[r, c1:c2] = s

    # filename = filename + ".png"
    path = None

    if int(filename[:-4]) in iid_train:
        path = os.path.join(train_destination, filename)
    elif int(filename[:-4]) in iid_test:
        path = os.path.join(test_destination, filename)
    else:
        print("filename is incorrect!")

    np.save(path, seg)
    # matplotlib.image.imsave(path, seg) # Saves but pixels values in [0.1]


path = "/home/ubuntu/workspace/W-Net-Pytorch/datasets/BSDS300/human/color/"
dirs = list()
for dir, _, files in os.walk(path):
    for filename in files:
        filepath = os.path.join(dir, filename)
        print(filepath)
        convertAndSave(filepath, filename)
