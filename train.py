# train.py

import os
import time
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
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
    base_dir = "../mri_coupled_dataset"

    # train_data = "dataset/train_filenames.txt"

    # Initialize the datasets
    train_dataset = MRIDataset(base_dir=base_dir)

    # Select a random subset of 20 items
    subset_indices = np.random.choice(len(train_dataset), 20, replace=False)
    train_subset = Subset(train_dataset, subset_indices)

    # Create the data loaders for the subset
    train_loader = DataLoader(
        train_subset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    # dataset = create_dataset(opt)
    # dataloader = DataLoader(
    #     dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
    # )
    dataset_size = len(train_dataset)
    print(f"The number of training images = {dataset_size}")

    # Create visualizer
    visualizer = Visualizer(opt)

    # Optionally resume training
    if opt.continue_train:
        load_checkpoint(model, opt.checkpoint_dir, opt.which_epoch, str(opt.device))
        print(f"Loading checkpoint on device: {opt.device}")

    # Training loop
    total_iters = 0
    losses_dict_arr = {"SR Loss": [], "GDN Loss": [], "GAN Loss": []}
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        mri_vol_index = 0

        for i, data in enumerate(train_loader, 0):
            epoch_iter += 1
            for j in range(data[0].size(0)):
                mri_vol_index += 1
                low_res_image, high_res_image = data[0][j], data[1][j]

                if high_res_image.shape[2:] != low_res_image.shape[2:]:
                    print(
                        f"Mismatched shapes in batch {i}: HR shape {high_res_image.shape}, LR shape {low_res_image.shape}"
                    )
                    continue

                current_batch_size = len(data[0])
                total_iters += current_batch_size

                mri_vol = {"LR": low_res_image, "HR": high_res_image}

                model.set_input(mri_vol)  # Prepare input data by slicing the MRI volume

                # Process each slice in the current volume
                num_slices = len(model.lr_slices)
                for slice_index in range(num_slices):
                    lr_slice, hr_slice = model.get_slice_pair(slice_index)

                    # Forward, backward pass, and optimize with additional parameters
                    model.optimize_parameters(
                        lr_images=lr_slice,
                        hr_images=hr_slice,
                        lambda_tv=opt.lambda_tv,
                        alpha_blur=opt.alpha_blur,
                        angle=opt.angle,
                        translation=(opt.translation_x, opt.translation_y),
                        weight_sr=opt.weight_sr,
                        weight_disc=opt.weight_disc,
                        weight_gdn=opt.weight_gdn,
                        alpha_l1=opt.alpha_l1,
                        beta_ssim=opt.beta_ssim,
                        gamma_psnr=opt.gamma_psnr,
                    )

                    # Print loss information at the specified frequency
                    # if total_iters % opt.print_freq == 0:
                    #     losses = model.get_current_losses()
                    #     t_comp = (time.time() - epoch_start_time) / epoch_iter
                    #     visualizer.print_current_losses(
                    #         epoch, epoch_iter, losses, t_comp, slice_index + 1, j + 1
                    #     )

                # Save the latest model at the specified frequency
                # if total_iters % opt.save_latest_freq == 0:
                #     print(
                #         "Saving the latest model (epoch %d, total_iters %d)"
                #         % (epoch, total_iters)
                #     )
                #     # save_checkpoint(
                #     #     model, opt.checkpoint_dir, "latest", epoch, total_iters
                #     # )
                #     model.save_checkpoint(
                #         opt.checkpoint_dir_vol,
                #         ["sr epoch_%d" % epoch, "vgg_patchgan epoch_%d" % epoch],
                #         epoch,
                #         total_iters,
                #     )

                # Display visuals at the specified frequency of the slices of a certain MRI Volume
                # if total_iters % opt.display_freq == 0:
                # model.save_volume(epoch=epoch)

                total_loss_sr = model.get_total_loss_of_volume()["loss_sr"] / num_slices
                total_loss_gdn = (
                    model.get_total_loss_of_volume()["gdnLoss"] / num_slices
                )
                total_loss_gan = (
                    model.get_total_loss_of_volume()["loss_gan"] / num_slices
                )

                losses_dict_arr["SR Loss"].append(total_loss_sr)
                losses_dict_arr["GDN Loss"].append(total_loss_gdn)
                losses_dict_arr["GAN Loss"].append(total_loss_gan)

                # print(
                #     "Epoch {}/{} | Batch Index: {} | MRI Volume Index: {} | SR Loss: {:.3f} | GDN Loss: {:.3f} | GAN Loss: {:.3f} | Time Taken: {} sec".format(
                #         epoch,
                #         opt.n_epochs + opt.n_epochs_decay,
                #         epoch_iter,
                #         mri_vol_index,
                #         total_loss_sr,
                #         total_loss_gdn,
                #         total_loss_gan,
                #         int(time.time() - epoch_start_time),
                #     )
                # )

                losses = {
                    "sr": total_loss_sr,
                    "gdn": total_loss_gdn,
                    "gan": total_loss_gan,
                }

                visualizer.print_current_statistics(
                    epoch=epoch,
                    batch_index=epoch_iter,
                    mri_vol_index=mri_vol_index,
                    losses=losses,
                    time_taken=int(time.time() - epoch_start_time),
                    total_epochs=opt.n_epochs + opt.n_epochs_decay,
                )

        # Save the latest model at the specified frequency
        if epoch % opt.save_epoch_freq == 0:
            print("Saving the latest model at (epoch %d)" % (epoch))
            # save_checkpoint(
            #     model, opt.checkpoint_dir, "latest", epoch, total_iters
            # )
            model.save_checkpoint(
                opt.checkpoint_dir_vol,
                ["sr epoch_%d" % epoch, "vgg_patchgan epoch_%d" % epoch],
                epoch,
                total_iters,
            )

    # Determine the number of unique values in each loss array
    unique_counts = {k: len(set(v)) for k, v in losses_dict_arr.items()}
    max_unique_count = max(unique_counts.values())

    # Calculate total_iters based on the max number of unique values and batch size
    total_iters_for_plots = list(
        range(0, max_unique_count * opt.batch_size, opt.batch_size)
    )

    visualizer.plot_and_save_losses(
        output_path=opt.plots_out_dir,
        total_iters=total_iters_for_plots,
        losses_dict_arr=losses_dict_arr,
    )

    model.save_final_models()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        # Optionally add code to handle specific exceptions and perform cleanup
