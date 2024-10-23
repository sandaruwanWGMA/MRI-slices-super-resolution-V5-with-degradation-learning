import os
import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(model, checkpoint_dir, filename, epoch, total_iters):
    """
    Saves the current state of the model and optimizer to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        checkpoint_dir (str): Directory path to save the checkpoint.
        filename (str): Filename to save the checkpoint.
        epoch (int): Current epoch of training.
        total_iters (int): Current iteration of training across all epochs.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{filename}.pth")
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "epoch": epoch,
    #         "total_iters": total_iters,
    #     },
    #     checkpoint_path,
    # )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "total_iters": total_iters,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_dir, filename, device):
    """
    Loads the model and optimizer states from a checkpoint file.

    Args:
        model (torch.nn.Module): The model for which the state will be loaded.
        optimizer (torch.optim.Optimizer): The optimizer for which the state will be loaded.
        checkpoint_dir (str): Directory path from where to load the checkpoint.
        filename (str): Filename to load the checkpoint from.
        device (torch.device or str): The device to load the checkpoint to.

    Returns:
        int: The epoch from which training should resume.
        int: The iteration from which training should resume.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"{filename}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Ensure device is compatible with torch.load
    if isinstance(device, torch.device):
        device = str(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    total_iters = checkpoint["total_iters"]
    print(
        f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}, iteration {total_iters}"
    )
    return epoch, total_iters
