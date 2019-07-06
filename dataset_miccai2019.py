import numpy as np
import os
import os.path
import SimpleITK as sitk

import torch.utils.data as data
import torch


def make_dataset(root):
    return [os.path.join(root, img_name) for img_name in os.listdir(os.path.join(root))]


class ImageFolder_miccai2019(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample_path = self.imgs[index]
        volume_path = os.path.join(sample_path, 'data.nii.gz')
        gt_path = os.path.join(sample_path, 'label.nii.gz')

        itk_CT = sitk.ReadImage(volume_path)
        itk_gt = sitk.ReadImage(gt_path)
        torch_CT = self._img_transfor(itk_CT)
        torch_gt = self._img_transfor(itk_gt)

        # if self.joint_transform is not None:
        #     img, target = self.joint_transform(itk_CT, itk_gt)

        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return torch_CT, torch_gt

    def __len__(self):
        return len(self.imgs)

    def _img_transfor(self, itk):
        img_arr = sitk.GetArrayFromImage(itk).astype(np.float32)
        torch_itk = torch.from_numpy(img_arr)
        return torch_itk

