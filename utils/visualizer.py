import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.utils as vutils
import torch


class Visualizer:
    def __init__(self, opt):
        """Initialize the Visualizer with options from the training configuration.

        Args:
            opt: An object containing configuration options, possibly from an ArgumentParser.
        """
        self.opt = opt
        self.image_dir = os.path.join(opt.checkpoint_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        # Set up matplotlib specifics
        plt.ion()  # Turn on interactive mode
        self.plots = {}

    def display_current_results(self, visuals, epoch, save_result):
        """Display or save current results.

        Args:
            visuals (dict): A dictionary containing images to display or save.
            epoch (int): Current epoch number for labeling purposes.
            save_result (bool): If True, saves the visuals to files.
        """
        for label, image_tensor in visuals.items():
            if save_result:
                image_numpy = self.tensor2im(image_tensor)
                save_path = os.path.join(self.image_dir, f"{label}_epoch_{epoch}.png")
                plt.imsave(save_path, image_numpy, format="png")
            if label not in self.plots:
                self.plots[label] = plt.figure(figsize=(8, 8))
            plt.figure(self.plots[label].number)
            plt.imshow(self.tensor2im(image_tensor))
            plt.title(f"{label} at Epoch {epoch}")
            plt.draw()
            plt.pause(0.001)

    def print_current_losses(
        self, epoch, counter, losses, time_per_batch, slice_index, mri_vol
    ):
        """Print current losses on the console.

        Args:
            epoch (int): Current epoch number.
            counter (int): Batch counter relative to the start of the epoch.
            losses (dict): A dictionary of losses.
            time_per_batch (float): Time taken for the current batch.
            slice_index (int): Index of the current slice.
        """
        message = f"(Epoch: {epoch}, Batch: {counter}, MRI Volume: {mri_vol}, Slice: {slice_index}) "
        message += ", ".join([f"{k}: {v:.3f}" for k, v in losses.items()])
        message += f", Time/Batch: {time_per_batch:.3f}"
        print(message)

    def tensor2im(self, image_tensor, imtype=np.uint8):
        """Convert a tensor to an image numpy array of type imtype.

        Args:
            image_tensor (torch.Tensor): The tensor to convert.
            imtype (type): The numpy type to convert to.

        Returns:
            numpy array of type imtype.
        """
        if isinstance(image_tensor, torch.Tensor):
            image_numpy = image_tensor.cpu().float().numpy()
            if image_numpy.shape[0] == 1:
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            return image_numpy.astype(imtype)
        else:
            return image_tensor
