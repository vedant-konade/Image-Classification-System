import os
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url

def get_transforms():
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                            std=[0.2470,0.2435,0.2616]),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                            std=[0.2470,0.2435,0.2616]),
    ])
    return train_tfms, test_tfms


CIFAR_MIRROR = "https://www.cs.toronto.edu/~kriz/"

def download_cifar10_fast(data_dir):
    base = "cifar-10-python.tar.gz"
    path = os.path.join(data_dir, base)
    if not os.path.exists(path):
        print("🔽 Downloading CIFAR-10 from fast mirror...")
        download_url(CIFAR_MIRROR + base, data_dir)

def get_loaders(data_dir="data", batch_size=128, seed=42):
    # try fast mirror first (speeds up download in some regions)
    download_cifar10_fast(data_dir)
    g = torch.Generator().manual_seed(seed)
    train_tfms, test_tfms = get_transforms()

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)

    # Split train into 70/15/15: from 50k, get 35k/7.5k/7.5k
    n_total = len(full_train)  # 50k
    n_train = int(0.70 * n_total)  # 35000
    n_val   = int(0.15 * n_total)  # 7500
    n_hold  = n_total - n_train - n_val   # 7500

    trainset, valset, _holdout = random_split(full_train, [n_train, n_val, n_hold], generator=g)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()
