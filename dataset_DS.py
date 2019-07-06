import numpy as np
import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(root):
    return [(os.path.join(root, 'us', img_name), os.path.join(root, 'seg', img_name),
            os.path.join(root, 'fnd_mean', img_name), os.path.join(root, 'fpd_mean', img_name)) for img_name in
            os.listdir(os.path.join(root, 'seg'))]


class ImageFolder_DS(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, fnd_path, fpd_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        fnd = Image.open(fnd_path).convert('L')
        fpd = Image.open(fpd_path).convert('L')
        if self.joint_transform is not None:
            img, target, fnd, fpd = self.joint_transform(img, target, fnd, fpd)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            fnd = self.target_transform(fnd)
            fpd = self.target_transform(fpd)

        return img, target, fnd, fpd

    def __len__(self):
        return len(self.imgs)
