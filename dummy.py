# train.py

import os
import time
import torch
from torch.utils.data import DataLoader
from model.create_model import create_model

# from data import create_dataset
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
from utils.checkpointing import save_checkpoint, load_checkpoint

from data.dataloader import MRIDataset


def main():
    # Parse options
    opt = TrainOptions().parse()

    print(f"Loading checkpoint on device: {opt.device}")

    # Create a model based on the options
    model = create_model(opt)

    # Base directory
    base_dir = "mri_coupled_dataset"

    # Initialize the datasets
    train_dataset = MRIDataset(base_dir=base_dir, transform=None)

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    print("We are good !!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        # Optionally add code to handle specific exceptions and perform cleanup
