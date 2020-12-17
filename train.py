import random

import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

from data import CIFAR10DataModule


@hydra.main(config_name="train")
def main(cfg: DictConfig):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    model = None
    datamodule = CIFAR10DataModule(
        cfg.dataset_dir,
        cfg.train_pct,
        cfg.batch_size,
        cfg.shuffle,
        cfg.num_workers,
        cfg.pin_memory,
        cfg.seed,
    )
    datamodule.prepare(cfg.stage)
    datamodule.setup(cfg.stage)
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
