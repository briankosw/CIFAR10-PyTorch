import os
import pickle
from typing import Tuple

import numpy as np


def calculate_dataset_statistics(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the mean and standard deviation of the CIFAR10 dataset.

    Arguments:
        dataset_dir: the path to the dataset directory

    Returns:
        a channel-wise mean and standard deviation
    """
    batches = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    images = []
    for batch in batches:
        filepath = os.path.join(dataset_dir, batch)
        with open(filepath, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            data = {k.decode(): v for k, v in data.items()}
            images.append(data["data"])
    images = np.concatenate(images).reshape(-1, 3, 32, 32) / 255.0
    return np.mean(images, axis=(0, 2, 3)), np.std(images, axis=(0, 2, 3))
