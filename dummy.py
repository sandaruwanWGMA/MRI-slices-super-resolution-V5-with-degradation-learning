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

    visualizer = Visualizer(opt)
    unique_values = 8000
    losses_dict_arr = {
        "sr_loss_arr": [x * 0.1 for x in range(unique_values)],
        "gdn_loss_arr": [x * 0.2 for x in range(unique_values)],
        "gan_loss_arr": [x * 0.3 for x in range(unique_values)],
    }

    # Determine the number of unique values in each loss array
    unique_counts = {k: len(set(v)) for k, v in losses_dict_arr.items()}
    max_unique_count = max(
        unique_counts.values()
    )  # Find the maximum number of unique values

    # Calculate total_iters based on the max number of unique values and batch size
    total_iters = list(range(0, max_unique_count * opt.batch_size, opt.batch_size))

    visualizer.plot_and_save_losses(
        output_path=opt.plots_out_dir,
        total_iters=total_iters,
        losses_dict_arr=losses_dict_arr,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        # Optionally add code to handle specific exceptions and perform cleanup
