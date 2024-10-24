import torch
import os
import numpy as np
import nibabel as nib

from .SRUNet import SRUNet
from .degradation_network import DegradationNetwork
from .VGGStylePatchGAN import VGGStylePatchGAN

from utils.losses import (
    perceptual_quality_loss,
    GDNLoss,
    perceptual_adversarial_loss,
    discriminator_loss,
)


class SuperResolutionModel:
    def __init__(self, opt):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and opt.gpu_ids[0] != -1 else "cpu"
        )

        # Initialize the models based on the configuration

        self.sr_unet = SRUNet(
            image_size=opt.image_size,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            freeze_encoder=opt.freeze_encoder,
            unfreeze_layers=opt.unfreeze_layers,
        ).to(
            self.device
        )  # Move SRUNet model to the correct device

        self.degradation_network = DegradationNetwork(image_size=opt.image_size).to(
            self.device
        )
        self.vgg_patch_gan = VGGStylePatchGAN(patch_size=opt.patch_size).to(self.device)

        # Optimizers for SRUNet and VGGStylePatchGAN only, since DegradationNetwork is not trained
        self.optimizer_sr = torch.optim.Adam(self.sr_unet.parameters(), lr=opt.lr)
        self.optimizer_gan = torch.optim.Adam(
            self.vgg_patch_gan.parameters(), lr=opt.lr
        )

        # Placeholder for storing input data
        self.lr_slices = []
        self.hr_slices = []

        self.current_losses = {}
        self.current_visuals = {"SR": [], "HR": []}

        self.loss_of_total_volume = {"loss_sr": 0, "gdnLoss": 0, "loss_gan": 0}

    def set_input(self, data):
        """
        Prepares the input data by extracting slices from a 3D MRI volume.
        """
        # Clear previous slice data
        self.lr_slices.clear()
        self.hr_slices.clear()

        self.loss_of_total_volume["loss_sr"] = 0
        self.loss_of_total_volume["loss_gan"] = 0
        self.loss_of_total_volume["gdnLoss"] = 0
        # Ensure data has the right structure and load LR and HR volumes
        lr_volume = (
            data["LR"].to(self.device).squeeze(0)
        )  # Expected shape: (150, 256, 256)
        hr_volume = (
            data["HR"].to(self.device).squeeze(0)
        )  # Expected shape: (150, 256, 256)

        # Check the dimensions of the input data
        assert (
            lr_volume.shape == hr_volume.shape
        ), "LR and HR volumes must have the same shape"
        num_slices, height, width = lr_volume.shape

        # Process each slice individually
        for i in range(num_slices):
            # Extract each slice and add a channel dimension (assumed to be grayscale images)
            lr_slice = lr_volume[i][None, :, :]  # Shape: (1, 256, 256)
            hr_slice = hr_volume[i][None, :, :]  # Shape: (1, 256, 256)

            # Convert slices to tensors and move to the appropriate device
            lr_tensor = (
                torch.tensor(lr_slice, dtype=torch.float32)
                .clone()
                .detach()
                .to(self.device)
                .unsqueeze(0)
            )  # Shape: (1, 1, 256, 256)
            hr_tensor = (
                torch.tensor(hr_slice, dtype=torch.float32)
                .clone()
                .detach()
                .to(self.device)
                .unsqueeze(0)
            )  # Shape: (1, 1, 256, 256)

            # Append to the slice list
            self.lr_slices.append(lr_tensor)
            self.hr_slices.append(hr_tensor)

    def get_slice_pair(self, index):
        """
        Retrieves the LR and HR tensors for a specific slice index.

        Args:
            index (int): The index of the slice to retrieve.

        Returns:
            tuple: A tuple (lr_slice, hr_slice) containing the LR and HR slice tensors.
        """
        if index < 0 or index >= len(self.lr_slices):
            raise IndexError("Slice index out of range")

        return self.lr_slices[index], self.hr_slices[index]

    def optimize_parameters(
        self,
        lr_images,
        hr_images,
        lambda_tv,
        alpha_blur,
        angle,
        translation,
        weight_sr,
        weight_disc,
        weight_gdn,
        alpha_l1,
        beta_ssim,
        gamma_psnr,
    ):
        # Normalize images before feeding to the networks

        # Step 1: Check the min and max pixel values in the images
        min_val_lr = torch.min(lr_images)
        max_val_lr = torch.max(lr_images)

        min_val_hr = torch.min(hr_images)
        max_val_hr = torch.max(hr_images)

        # Step 2: Normalize the images based on the min and max values (scaling to [-1, 1])
        # Handle case where min and max are the same (constant image)
        if max_val_lr == min_val_lr:
            # Step 1: Standardize the image (mean 0, variance 1)
            mean_lr = torch.mean(lr_images)
            std_lr = torch.std(lr_images)

            # Ensure std is not zero to avoid division by zero
            if std_lr == 0:
                lr_images_normalized = torch.zeros_like(
                    lr_images
                )  # Handle constant image case
            else:
                # Standardize to mean 0, variance 1
                lr_images_standardized = (lr_images - mean_lr) / std_lr

                # Step 2: Scale to range [-1, 1]
                # Find the max absolute value to scale to [-1, 1]
                max_val = torch.max(torch.abs(lr_images_standardized))
                lr_images_normalized = lr_images_standardized / max_val

            # Clamp values to ensure they are exactly within [-1, 1] if any values drift
            lr_images_normalized = torch.clamp(lr_images_normalized, min=-1.0, max=1.0)

        else:
            lr_images_normalized = (
                2 * (lr_images - min_val_lr) / (max_val_lr - min_val_lr) - 1
            )

        if max_val_hr == min_val_hr:
            # Step 1: Standardize the image (mean 0, variance 1)
            mean_hr = torch.mean(hr_images)
            std_hr = torch.std(hr_images)

            # Ensure std is not zero to avoid division by zero
            if std_hr == 0:
                hr_images_normalized = torch.zeros_like(
                    hr_images
                )  # Handle constant image case
            else:
                # Standardize to mean 0, variance 1
                hr_images_standardized = (hr_images - mean_hr) / std_hr

                # Step 2: Scale to range [-1, 1]
                # Find the max absolute value to scale to [-1, 1]
                max_val = torch.max(torch.abs(hr_images_standardized))
                hr_images_normalized = hr_images_standardized / max_val

            # Clamp values to ensure they are exactly within [-1, 1] if any values drift
            hr_images_normalized = torch.clamp(hr_images_normalized, min=-1.0, max=1.0)

        else:
            hr_images_normalized = (
                2 * (hr_images - min_val_hr) / (max_val_hr - min_val_hr) - 1
            )

        # Ensure that the normalized values are indeed within the range [-1, 1]
        lr_images_normalized = torch.clamp(lr_images_normalized, min=-1.0, max=1.0)
        hr_images_normalized = torch.clamp(hr_images_normalized, min=-1.0, max=1.0)

        # Step 3: Re-run the assertion to verify
        assert torch.all(lr_images_normalized >= -1) and torch.all(
            lr_images_normalized <= 1
        ), "lr_images_normalized has values outside [-1, 1]"
        assert torch.all(hr_images_normalized >= -1) and torch.all(
            hr_images_normalized <= 1
        ), "hr_images_normalized has values outside [-1, 1]"

        # Zero gradients for both optimizers
        self.optimizer_sr.zero_grad()
        self.optimizer_gan.zero_grad()

        # Forward pass through SRUNet
        sr_output = self.sr_unet(lr_images_normalized)
        blur_kernel = self.degradation_network(sr_output, angle, translation)

        # Forward pass through VGGStylePatchGAN
        real_pred = self.vgg_patch_gan(hr_images_normalized)
        fake_pred = self.vgg_patch_gan(sr_output)

        # Calculate losses
        loss_sr = perceptual_quality_loss(
            sr_output,
            hr_images_normalized,
            alpha=alpha_l1,
            beta=beta_ssim,
            gamma=gamma_psnr,
        )
        loss_gan = discriminator_loss(real_preds=real_pred, fake_preds=fake_pred)
        loss_gdn = GDNLoss(sr_output, lr_images, blur_kernel, lambda_tv, alpha_blur)

        # Total loss
        total_loss = (
            weight_sr * loss_sr + weight_disc * loss_gan + weight_gdn * loss_gdn
        )

        # Backward pass
        total_loss.backward()

        # Update parameters
        self.optimizer_sr.step()
        self.optimizer_gan.step()

        # Output the losses in a dictionary
        loss_results = {
            "loss_sr": loss_sr.item(),
            "loss_gdn": loss_gdn.item(),
            "loss_gan": loss_gan.item(),
        }

        self.loss_of_total_volume["loss_sr"] += loss_results["loss_sr"]
        self.loss_of_total_volume["loss_gan"] += loss_results["loss_gan"]
        self.loss_of_total_volume["gdnLoss"] += loss_results["loss_gdn"]

        # Store the current losses
        self.current_losses = loss_results

        return loss_results

    def get_total_loss_of_volume(self):
        return self.loss_of_total_volume

    def get_current_losses(self):
        """
        Returns the losses for the currently processed slice.

        Returns:
            dict: A dictionary containing the current losses: 'loss_sr', 'loss_gdn', and 'loss_gan'.
        """

        # Return the most recently stored losses
        return self.current_losses

    def compute_visuals(self):
        if "SR" not in self.current_visuals or len(self.current_visuals["SR"]) != len(
            self.lr_slices
        ):
            print(
                "SR visuals are not fully computed; make sure to run optimize_parameters for all slices."
            )

    def save_volume(self, epoch):
        """
        Assembles and saves the current LR and SR slices into 3D MRI volumes in NIfTI format.

        Args:
            epoch (int): The current epoch number for folder naming and file naming.
        """
        # Create folder for saving the volumes
        save_folder = os.path.join("results", f"epoch_{epoch}")
        os.makedirs(save_folder, exist_ok=True)

        print("sv 1")

        # Assemble LR and SR slices into 3D volumes
        lr_volume = np.stack(
            [lr_slice.squeeze().cpu().numpy() for lr_slice in self.lr_slices], axis=0
        )

        print("sv 2")

        sr_volume = np.stack(
            [self.current_visuals["SR"].squeeze() for _ in range(len(self.lr_slices))],
            axis=0,
        )
        print("sv 3")

        # Save LR and SR volumes as .nii files
        lr_img = nib.Nifti1Image(lr_volume, affine=np.eye(4))
        sr_img = nib.Nifti1Image(sr_volume, affine=np.eye(4))

        print("sv 4")

        lr_file_path = os.path.join(save_folder, f"epoch{epoch}_LR.nii")
        sr_file_path = os.path.join(save_folder, f"epoch{epoch}_HR.nii")

        print("sv 5")

        nib.save(lr_img, lr_file_path)
        print("sv 6")
        nib.save(sr_img, sr_file_path)

        print(f"Saved LR volume to {lr_file_path}")
        print(f"Saved SR volume to {sr_file_path}")

    def save_final_models(self):
        """
        Saves the final trained SRUNet and VGGStylePatchGAN models to disk after training completion.
        """
        # Specify the folder and file paths for saving the models
        save_folder = "final_models"
        os.makedirs(save_folder, exist_ok=True)
        sr_unet_file_path = os.path.join(save_folder, "SRUNet_final.pth")
        vgg_patch_gan_file_path = os.path.join(
            save_folder, "VGGStylePatchGAN_final.pth"
        )

        # Save the models
        torch.save(self.sr_unet.state_dict(), sr_unet_file_path)
        torch.save(self.vgg_patch_gan.state_dict(), vgg_patch_gan_file_path)

        print(f"Saved final SRUNet model to {sr_unet_file_path}")
        print(f"Saved final VGGStylePatchGAN model to {vgg_patch_gan_file_path}")

    def save_checkpoint(self, checkpoint_dir, filename, epoch, total_iters):
        """
        Saves the partially trained SRUNet and VGGStylePatchGAN models to disk.
        """
        print("Start saving partial models...")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path_sr = os.path.join(checkpoint_dir, f"{filename[0]}.pth")
        checkpoint_path_vgg_patch_gan = os.path.join(
            checkpoint_dir, f"{filename[1]}.pth"
        )

        # Save the models
        torch.save(
            {
                "model_state_dict": self.sr_unet.state_dict(),
                "epoch": epoch,
                "total_iters": total_iters,
            },
            checkpoint_path_sr,
        )
        torch.save(
            {
                "model_state_dict": self.vgg_patch_gan.state_dict(),
                "epoch": epoch,
                "total_iters": total_iters,
            },
            checkpoint_path_vgg_patch_gan,
        )
        # torch.save(self.sr_unet.state_dict(), checkpoint_path_sr)
        # torch.save(self.vgg_patch_gan.state_dict(), checkpoint_path_vgg_patch_gan)

        print(f"Checkpoint saved to {checkpoint_path_sr}")
        print(f"Checkpoint saved to {checkpoint_path_vgg_patch_gan}")
