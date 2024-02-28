import math
import os

import numpy as np
import torch
from torchvision.datasets import VisionDataset
from PIL import Image


class OODDataset(torch.utils.data.Dataset):
    def __init__(self, in_dataset, ood_dataset):
        self.in_dataset = in_dataset
        self.n_in = len(in_dataset)
        self.ood_dataset = ood_dataset
        self.n_ood = len(ood_dataset)

    def __getitem__(self, i):
        if i < self.n_in:
            # Fetch in-distribution data
            input, target = self.in_dataset[i]
            return input, 0
        else:
            # Fetch out-of-distribution data
            input, target = self.ood_dataset[i - self.n_in]
            return input, 1

    def __len__(self):
        return self.n_in + self.n_ood


class CorruptedCIFAR10(VisionDataset):
    corruption_types = [
        'brightness',
        'contrast',
        'defocus_blur',
        'elastic_transform',
        'fog',
        'frost',
        'gaussian_blur',
        'gaussian_noise',
        'glass_blur',
        'impulse_noise',
        'jpeg_compression',
        'motion_blur',
        'pixelate',
        'saturate',
        'shot_noise',
        'snow',
        'spatter',
        'speckle_noise',
        'zoom_blur']
    max_intensity = 5
    data_dir = 'CIFAR-10-C'

    def __init__(self, root, skew_intensity, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert skew_intensity in [1, 2, 3, 4, 5]
        self.targets = np.tile(np.load(os.path.join(root, self.data_dir, 'labels.npy'))[:10000], len(self.corruption_types))
        self.data = []
        for corruption_type in self.corruption_types:
            self.data.append(np.load(os.path.join(root, self.data_dir, corruption_type + '.npy'))[10000*(skew_intensity-1):10000*skew_intensity])

        self.data = np.concatenate(self.data, axis=0)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class CorruptedCIFAR100(CorruptedCIFAR10):
    data_dir = 'CIFAR-100-C'


def compute_ece(confs, targets, bins=10):
    device = confs.device
    bin_boundaries = torch.linspace(0, 1, bins + 1, dtype=torch.float, device=device)

    confs, preds = confs.max(dim=1)
    confs = confs.float()
    accs = preds.eq(targets).float()

    # computes bins
    acc_bin = torch.zeros(bins, device=device)
    prob_bin = torch.zeros(bins, device=device)
    count_bin = torch.zeros(bins, device=device)

    indices = torch.bucketize(confs, bin_boundaries) - 1
    count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confs))
    prob_bin.scatter_add_(dim=0, index=indices, src=confs)
    prob_bin = torch.nan_to_num(prob_bin / count_bin)
    acc_bin.scatter_add_(dim=0, index=indices, src=accs)
    acc_bin = torch.nan_to_num(acc_bin / count_bin)
    prop_bin = count_bin / count_bin.sum()

    ece = torch.sum(torch.abs(acc_bin - prob_bin) * prop_bin)
    return ece 


class ExpectedCalibrationError():
    def __init__(self, device, bins=10):
        self.bins = bins
        self.bin_boundaries = torch.linspace(0, 1, bins + 1, dtype=torch.float, device=device)
        self.acc_bin = torch.zeros(bins, device=device)
        self.prob_bin = torch.zeros(bins, device=device)
        self.count_bin = torch.zeros(bins, device=device)

    def update(self, confs, targets):
        confs, preds = confs.max(dim=1)
        confs = confs.float()
        accs = preds.eq(targets).float()

        # if confs is 1, causes index errors in the following scatter_add
        indices = torch.bucketize(confs, self.bin_boundaries).clamp(min=1, max=self.bins) - 1
        self.count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confs))
        self.prob_bin.scatter_add_(dim=0, index=indices, src=confs)
        self.acc_bin.scatter_add_(dim=0, index=indices, src=accs)

        _prob_bin = torch.nan_to_num(self.prob_bin / self.count_bin)
        _acc_bin = torch.nan_to_num(self.acc_bin / self.count_bin)
        prop_bin = self.count_bin / self.count_bin.sum()

        ece = torch.sum(torch.abs(_acc_bin - _prob_bin) * prop_bin)
        return ece


def causal_mask(width, height, starting_point):
    row_grid, col_grid = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    mask = np.logical_or(
        row_grid < starting_point[0],
        np.logical_and(row_grid == starting_point[0], col_grid <= starting_point[1]))
    return torch.tensor(mask)

def conv_mask(width, height, include_center=False):
    return 1.0 * causal_mask(width, height, starting_point=(width//2, height//2 + include_center - 1))

def weight_mask(in_channels, kernel_size):
    conv_mask_with_center = conv_mask(kernel_size, kernel_size, include_center=True)
    conv_mask_no_center = conv_mask(kernel_size, kernel_size, include_center=False)

    mask = torch.zeros(in_channels, in_channels, kernel_size, kernel_size)
    for i in range(in_channels):
        for j in range(in_channels):
            mask[i][j] = conv_mask_no_center if j >= i else conv_mask_with_center
    return mask