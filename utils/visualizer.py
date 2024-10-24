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

    def plot_and_save_losses(self, output_path, total_iters, losses_dict_arr):
        """
        Plot each type of loss against the total iterations and save the plots to the specified directory.

        Args:
            output_path (str): The directory path where the plots will be saved.
            total_iters (list): An array of iteration numbers.
            losses_dict_arr (dict): A dictionary where each key is a loss name and each value is an array of loss values.
        """
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Figure size adjustment for clarity
        plt.figure(figsize=(12, 6))  # Wider plot to better distribute data points

        # Iterate through each loss array in the dictionary
        for loss_name, loss_values in losses_dict_arr.items():
            plt.plot(total_iters, loss_values, label=loss_name)
            plt.xlabel("Iteration Number")
            plt.ylabel("Loss Value")
            plt.title(f"{loss_name} Loss Over Iterations")
            plt.legend()
            plt.grid(True)

            # Managing tick density
            plt.xticks(rotation=45)  # Rotate x-axis labels to avoid overlap
            plt.gca().xaxis.set_major_locator(
                plt.MaxNLocator(20)
            )  # Limit number of x-axis ticks

            # Save the plot to the specified output directory
            file_path = os.path.join(output_path, f"{loss_name}.png")
            plt.savefig(
                file_path, bbox_inches="tight"
            )  # Ensure no clipping of tick labels
            plt.close()  # Close the plot to free up memory

            print(f"Plot saved: {file_path}")

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

    def print_current_statistics(
        self, epoch, batch_index, mri_vol_index, losses, time_taken, total_epochs
    ):
        """Print current training statistics on the console.

        Args:
            epoch (int): Current epoch number.
            batch_index (int): Batch index relative to the start of the epoch.
            mri_vol_index (int): MRI volume index.
            losses (dict): A dictionary containing 'sr', 'gdn', and 'gan' losses.
            epoch_start_time (float): Timestamp when the epoch started.
            total_epochs (int): Total number of epochs, including both training and decay epochs.
        """
        sr_loss = losses.get("sr", 0)
        gdn_loss = losses.get("gdn", 0)
        gan_loss = losses.get("gan", 0)

        message = (
            f"Epoch {epoch}/{total_epochs} | Batch Index: {batch_index} | "
            f"MRI Volume Index: {mri_vol_index} | SR Loss: {sr_loss:.3f} | "
            f"GDN Loss: {gdn_loss:.3f} | GAN Loss: {gan_loss:.3f} | "
            f"Time Taken: {time_taken} sec"
        )
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
