import numpy as np
import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(root):
    return [(os.path.join(root, 'us', img_name), os.path.join(root, 'seg', img_name),
             (os.path.join(root, 'counter', img_name))) for img_name in
            os.listdir(os.path.join(root, 'seg'))]


class ImageFolder_multi_task(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, counter_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        counter = Image.open(counter_path)
        if self.joint_transform is not None:
            img, target, counter = self.joint_transform(img, target, counter)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            counter = self.target_transform(counter)

        return img, target, counter

    def __len__(self):
        return len(self.imgs)
