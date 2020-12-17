__all__ = ["CIFAR10DataModule"]

from itertools import chain
import os
from typing import Any, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

NUM_CLASSES = 10
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        train_pct: float,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: Optional[int] = None,
    ):
        self.dataset_dir = dataset_dir
        self.train_pct = train_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def prepare(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            CIFAR10(self.dataset_dir, train=True, download=True)
        if stage == "evaluate" or stage is None:
            CIFAR10(self.dataset_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset = CIFAR10(
                self.dataset_dir,
                train=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                    ]
                ),
            )
            num_train = int(len(dataset) * self.train_pct)
            num_val = len(dataset) - num_train
            generator = None
            if self.seed is not None:
                generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                dataset, lengths=[num_train, num_val], generator=generator
            )

        if stage == "evaluate" or stage is None:
            self.test_dataset = CIFAR10(
                self.dataset_dir,
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                    ]
                ),
            )

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
