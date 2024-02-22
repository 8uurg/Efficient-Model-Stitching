"""
Note: this file mostly relates to the CIFAR dataset, and hence was not actually used within the work itself.
However, our problem statements file includes a reference to this file, and therefore was left be.
"""

from pathlib import Path
import polars as pl
import torch
from torch.utils.data import random_split
# reminder: use v2 for segmentation & similar vision tasks
from torchvision.transforms import Normalize, ToTensor, Compose, RandomHorizontalFlip, RandomErasing, RandomResizedCrop
import math
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

# Constants commonly used for image preprocessing
mean_norm = 128
mean_std = (255 - 0) / math.sqrt(12)

datasets_folder = "<add-dataset-folder>" # Path(__file__).parent.parent / 'datasets'

def get_cifar_transform(augment=False, preaug=None):
    transforms = [
        ToTensor(),
    ]

    if preaug is not None:
        transforms += preaug

    transforms += [
        Normalize(mean=0.5, std=0.25)
    ]

    if augment != False:
        transforms += [
            RandomHorizontalFlip(),
            RandomResizedCrop((32, 32), scale=(0.95, 1.0), antialias=True),
            RandomErasing(scale=(0.01,0.10)),
        ]
    return Compose(transforms)
    

def load_CIFAR100(download=False, augment=False):
    transform_cifar100 = get_cifar_transform(augment=augment)
    dataset_cifar100_train_full = CIFAR100(
        datasets_folder, download=download, train=True, transform=transform_cifar100 # type: ignore
    )
    dataset_cifar100_test = CIFAR100(
        datasets_folder, download=download, train=False, transform=transform_cifar100 # type: ignore
    )
    rng_split = torch.manual_seed(42)
    dataset_train_full = dataset_cifar100_train_full
    dataset_cifar100_train, dataset_cifar100_validation = random_split(
        dataset_train_full,
        [round(0.9 * len(dataset_train_full)), round(0.1 * len(dataset_train_full))],
        generator=rng_split,
    )
    return dataset_cifar100_train, dataset_cifar100_validation, dataset_cifar100_test


def load_CIFAR10(download=False, augment=False):
    # inferred from train statistics
    transform_cifar10 = get_cifar_transform(augment=augment)
    dataset_cifar10_train_full = CIFAR10(
        datasets_folder, download=download, train=True, transform=transform_cifar10 # type: ignore
    )
    dataset_cifar10_test = CIFAR10(
        datasets_folder, download=download, train=False, transform=transform_cifar10 # type: ignore
    )
    rng_split = torch.manual_seed(42)
    dataset_train_full = dataset_cifar10_train_full
    dataset_cifar10_train, dataset_cifar10_validation = random_split(
        dataset_train_full,
        [round(0.9 * len(dataset_train_full)), round(0.1 * len(dataset_train_full))],
        generator=rng_split,
    )
    return dataset_cifar10_train, dataset_cifar10_validation, dataset_cifar10_test

def load_CIFAR10_preaug(preaug, download=False, augment=False):
    # inferred from train statistics
    transform_cifar10 = get_cifar_transform(augment=augment, preaug=preaug)

    dataset_cifar10_train_full = CIFAR10(
        datasets_folder, download=download, train=True, transform=transform_cifar10 # type: ignore
    )
    dataset_cifar10_test = CIFAR10(
        datasets_folder, download=download, train=False, transform=transform_cifar10 # type: ignore
    )
    rng_split = torch.manual_seed(42)
    dataset_train_full = dataset_cifar10_train_full
    dataset_cifar10_train, dataset_cifar10_validation = random_split(
        dataset_train_full,
        [round(0.9 * len(dataset_train_full)), round(0.1 * len(dataset_train_full))],
        generator=rng_split,
    )
    return dataset_cifar10_train, dataset_cifar10_validation, dataset_cifar10_test


def load_MNIST(download=False):
    transform_mnist = Compose([ToTensor(), Normalize(mean=0, std=2 * mean_std)])
    dataset_mnist_train_full = MNIST(
        datasets_folder, download=download, train=True, transform=transform_mnist # type: ignore
    )
    dataset_mnist_test = MNIST(
        datasets_folder, download=download, train=False, transform=transform_mnist # type: ignore
    )
    rng_split = torch.manual_seed(42)
    dataset_train_full = dataset_mnist_train_full
    dataset_mnist_train, dataset_mnist_validation = random_split(
        dataset_train_full,
        [round(0.9 * len(dataset_train_full)), round(0.1 * len(dataset_train_full))],
        generator=rng_split,
    )
    return dataset_mnist_train, dataset_mnist_validation, dataset_mnist_test


def load_FashionMNIST(download=False):
    # sample based: mean = 0.2862, variance = 0.1241, std = 0.35228
    # domain based: mean = 0.5, variance = 1/12, std = 1/sqrt(12)
    transform_mnist = Compose([ToTensor(), Normalize(mean=0.5, std=1 / math.sqrt(12))])
    dataset_fashionmnist_train_full = FashionMNIST(
        datasets_folder, download=download, train=True, transform=transform_mnist # type: ignore
    )
    dataset_fashionmnist_test = FashionMNIST(
        datasets_folder, download=download, train=False, transform=transform_mnist # type: ignore
    )
    rng_split = torch.manual_seed(42)
    dataset_train_full = dataset_fashionmnist_train_full
    dataset_fashionmnist_train, dataset_fashionmnist_validation = random_split(
        dataset_train_full,
        [round(0.9 * len(dataset_train_full)), round(0.1 * len(dataset_train_full))],
        generator=rng_split,
    )
    return (
        dataset_fashionmnist_train,
        dataset_fashionmnist_validation,
        dataset_fashionmnist_test,
    )

class SplitCIFAR100Dataset(Dataset):
    
    def __init__(self, root, split_idx_or_test, transform=None):
        self.root = root
        self.transform = transform
        
        if isinstance(split_idx_or_test, int):
            self.train = True
            self.split_idx = split_idx_or_test
            self.data = pl.read_ipc(self.root / 'cifar-100-python-split' / f'split-{self.split_idx}.feather')
        else:
            self.train = False
            self.data = pl.read_ipc(self.root / 'cifar-100-python-split' / 'test.feather')

    def __getitem__(self, index):
        # Grab the requested sample from the dataframe & transform it to an appropriately shaped numpy array.
        row = self.data[index]
        image = row['data'].item().to_numpy().reshape(3, 32, 32)
        label = row['coarse_label'].item()
        # Transform, if a transformation is provided.
        if self.transform is not None:
            image = self.transform(image)
        # Return the transformed image and the label.
        return image, label
    
    def __len__(self):
        return self.data.shape[0]
    
def load_CIFAR100presplit(split_idx, augment=False):
    transform_cifar100 = get_cifar_transform(augment=augment)
    dataset_cifar100_train_presplit = SplitCIFAR100Dataset(
        datasets_folder, split_idx, transform=transform_cifar100 # type: ignore
    )
    dataset_cifar100_test = CIFAR100(
        datasets_folder, "test", transform=transform_cifar100 # type: ignore
    )
    rng_split = torch.manual_seed(42)
    dataset_train_full = dataset_cifar100_train_presplit
    dataset_cifar100_train, dataset_cifar100_validation = random_split(
        dataset_train_full,
        [round(0.9 * len(dataset_train_full)), round(0.1 * len(dataset_train_full))],
        generator=rng_split,
    )
    return dataset_cifar100_train, dataset_cifar100_validation, dataset_cifar100_test
