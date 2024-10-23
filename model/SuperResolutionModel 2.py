import torch
import os
import numpy as np
import nibabel as nib

from .SRUNet import SimpleNet
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
        # image_size=opt.image_size,
        # in_channels=opt.in_channels,
        # out_channels=opt.out_channels,
        # freeze_encoder=opt.freeze_encoder,
        self.sr_unet = SimpleNet().to(
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
        self.current_visuals = {}

    def set_input(self, data):
        """
        Prepares the input data by extracting slices from a 3D MRI volume.
        """
        # Clear previous slice data
        self.lr_slices.clear()
        self.hr_slices.clear()

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

    def optimize_parameters(self, lr_images, hr_images, lambda_tv, angle, translation):
        # Print input tensor shapes
        print(f"Initial lr_images shape: {lr_images.shape}")
        print(f"Initial hr_images shape: {hr_images.shape}")

        # Step 1: Forward pass through SRUNet
        sr_output = self.sr_unet(lr_images)
        # print("sr_output2: ", sr_output2)
        # sr_output = torch.rand(1, 1, 256, 256, requires_grad=True).to(self.device)
        # print("sr_output2: ", sr_output2)
        print(f"Shape of sr_output after SRUNet: {sr_output.shape}")

        # Store SR output for later visualization
        if "SR" not in self.current_visuals:
            self.current_visuals["SR"] = []

        self.current_visuals["SR"].append(sr_output.squeeze().cpu().detach().numpy())
        print(f"Stored sr_output shape in visuals: {sr_output.squeeze().shape}")

        # Step 2: Pass SRUNet output through DegradationNetwork to compute the feedback loss
        blur_kernel = self.degradation_network(sr_output, angle, translation)
        print(f"Shape of blur_kernel from DegradationNetwork: {blur_kernel.shape}")

        # Step 3: Prepare input for VGGStylePatchGAN

        # Forward pass through VGGStylePatchGAN
        real_pred = self.vgg_patch_gan(hr_images)
        fake_pred = self.vgg_patch_gan(sr_output)
        print(f"real_pred shape: {real_pred.shape}")
        print(f"fake_pred shape: {fake_pred.shape}")
        print("real_pred: ", real_pred)
        print("fake_pred: ", fake_pred)

        # Add checks before loss calculation
        if (real_pred < 0).any() or (fake_pred < 0).any():
            print("Error: Prediction values are negative where expected non-negative.")

        # Step 4: Calculate losses
        # Loss for SRUNet
        print(
            f"Before loss_sr computation, sr_output range: {sr_output.min()} to {sr_output.max()}, hr_images range: {hr_images.min()} to {hr_images.max()}"
        )

        hr_images = hr_images.clone()
        hr_min = hr_images.min()
        hr_max = hr_images.max()

        if hr_max > hr_min:
            hr_images_normalized = (hr_images - hr_min) / (hr_max - hr_min)
        else:
            hr_images_normalized = hr_images - hr_min  # Avoid in-place operation

        # Print to verify normalization
        print(f"Original hr_images range: {hr_min.item()} to {hr_max.item()}")
        print(
            f"Normalized hr_images range: {hr_images_normalized.min()} to {hr_images_normalized.max()}"
        )

        loss_sr = perceptual_quality_loss(
            sr_output, hr_images_normalized, alpha=1.0, beta=1.0, gamma=1.0
        )
        print(f"Loss SR: {loss_sr}")

        # Feedback loss from DegradationNetwork - affects SRUNet
        print(
            f"Before loss_gdn computation, sr_output range: {sr_output.min()} to {sr_output.max()}, lr_images range: {lr_images.min()} to {lr_images.max()}, blur_kernel range: {blur_kernel.min()} to {blur_kernel.max()}"
        )
        loss_gdn = GDNLoss(sr_output, lr_images, blur_kernel, lambda_tv)
        print(f"Loss GDN: {loss_gdn}")

        # Loss for VGGStylePatchGAN
        print(
            f"Before loss_gan computation, real_pred range: {real_pred.min()} to {real_pred.max()}, fake_pred range: {fake_pred.min()} to {fake_pred.max()}"
        )
        torch.autograd.set_detect_anomaly(True)
        loss_gan = perceptual_adversarial_loss(
            hr_images_normalized,
            sr_output,
            real_pred,
            fake_pred,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            delta=1.0,
        )

        # loss_gan = discriminator_loss(real_preds=real_pred, fake_preds=fake_pred)

        print(f"Loss GAN: {loss_gan}")

        # Total loss for SRUNet includes both perceptual and degradation feedback losses
        total_loss_sr = loss_sr + loss_gdn
        print(f"Total Loss SR: {total_loss_sr}")

        # Step 5: Backpropagation and optimization for SRUNet
        self.optimizer_sr.zero_grad()
        total_loss_sr.backward(
            retain_graph=True
        )  # Retain graph for subsequent backpropagation
        self.optimizer_sr.step()

        print("Backprop SR")

        # Backpropagation and optimization for VGGStylePatchGAN
        torch.autograd.set_detect_anomaly(True)

        self.optimizer_gan.zero_grad()
        print("Zero grad discriminator")
        print("BEFORE")
        for name, param in self.vgg_patch_gan.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(name, param.grad.norm().item())

        loss_gan.backward()
        print("AFTER")
        for name, param in self.vgg_patch_gan.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(name, param.grad.norm().item())
        self.optimizer_gan.step()

        print("Backprop VGG PatchGAN")

        # Calculate losses
        loss_results = {
            "loss_sr": loss_sr.item(),
            "loss_gdn": loss_gdn.item(),
            "loss_gan": loss_gan.item(),
        }

        # Store the current losses
        self.current_losses = loss_results
        print(f"Current losses: {self.current_losses}")

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

        # Assemble LR and SR slices into 3D volumes
        lr_volume = np.stack(
            [lr_slice.squeeze().cpu().numpy() for lr_slice in self.lr_slices], axis=0
        )
        sr_volume = np.stack(
            [self.current_visuals["SR"].squeeze() for _ in range(len(self.lr_slices))],
            axis=0,
        )

        # Save LR and SR volumes as .nii files
        lr_img = nib.Nifti1Image(lr_volume, affine=np.eye(4))
        sr_img = nib.Nifti1Image(sr_volume, affine=np.eye(4))

        lr_file_path = os.path.join(save_folder, f"epoch{epoch}_LR.nii")
        sr_file_path = os.path.join(save_folder, f"epoch{epoch}_HR.nii")

        nib.save(lr_img, lr_file_path)
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
