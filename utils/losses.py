import torch
import torch.nn as nn
import torch.nn.functional as F
import piq  # SSIM and PSNR


def perceptual_quality_loss(output, target, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Computes a weighted combination of L1, SSIM, and PSNR losses.

    Parameters:
        output (Tensor): Generated image.
        target (Tensor): Ground truth image.
        alpha (float): Weight for L1 loss.
        beta (float): Weight for SSIM loss.
        gamma (float): Weight for PSNR loss.

    Returns:
        Tensor: Weighted combined loss.
    """
    # L1 Loss (Mean Absolute Error)
    l1_loss = F.l1_loss(output, target)

    # SSIM Loss
    ssim_loss = 1 - piq.ssim(output, target, data_range=1.0)

    # PSNR Loss (lower PSNR corresponds to higher loss)
    psnr_value = piq.psnr(output, target, data_range=1.0)
    psnr_loss = 1 / (psnr_value + 1e-7)

    # Weighted sum of losses
    combined_loss = (alpha * l1_loss) + (beta * ssim_loss) + (gamma * psnr_loss)

    return combined_loss


def perceptual_quality_loss(output, target, alpha=0.5, beta=1.0, gamma=0.25):
    """
    Computes a weighted combination of L1, SSIM, and PSNR losses.

    Parameters:
        output (Tensor): Generated image.
        target (Tensor): Ground truth image.
        alpha (float): Weight for L1 loss.
        beta (float): Weight for SSIM loss.
        gamma (float): Weight for PSNR loss.

    Returns:
        Tensor: Weighted combined loss.
    """
    # L1 Loss (Mean Absolute Error)
    l1_loss = F.l1_loss(output, target)

    # Rescale from [-1, 1] to [0, 1]
    output = (output + 1) / 2
    target = (target + 1) / 2

    assert torch.all(
        (output >= 0) & (output <= 1)
    ), "Generated image has pixel values out of the [0, 1] range."
    assert torch.all(
        (target >= 0) & (target <= 1)
    ), "Real image has pixel values out of the [0, 1] range."

    # SSIM Loss
    ssim_loss = 1 - piq.ssim(output, target, data_range=1.0)

    # PSNR Loss (lower PSNR corresponds to higher loss)
    psnr_value = piq.psnr(output, target, data_range=1.0)

    psnr_loss = 1 / (psnr_value + 1e-7)

    # Weighted sum of losses
    combined_loss = (alpha * l1_loss) + (beta * ssim_loss) + (gamma * psnr_loss)

    return combined_loss


import torch
import torch.nn.functional as F
import piq  # For SSIM and PSNR


def discriminator_loss(real_preds, fake_preds):
    """
    Calculates the adversarial loss for the discriminator in a GAN.

    Parameters:
    - real_preds: the discriminator's predictions on real images.
    - fake_preds: the discriminator's predictions on fake (generated) images.

    Returns:
    - loss: the total adversarial loss for the discriminator.
    """
    # Loss for correctly identifying real images as real
    real_loss = F.binary_cross_entropy_with_logits(
        real_preds, torch.ones_like(real_preds)
    )

    # Loss for correctly identifying fake images as fake
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_preds, torch.zeros_like(fake_preds)
    )

    # Average these two losses
    loss = (real_loss + fake_loss) / 2

    return loss


def perceptual_adversarial_loss(
    real_images,
    generated_images,
    real_preds,
    fake_preds,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    delta=1.0,
):
    """
    Combines adversarial loss with perceptual losses (L1, SSIM, PSNR) for MRI super-resolution tasks.
    """
    # Clone tensors to prevent in-place modification issues
    real_preds = real_preds.clone()
    fake_preds = fake_preds.clone()
    real_images = real_images.clone()
    generated_images = generated_images.clone()

    # Adversarial Loss (BCE)
    adversarial_loss_real = F.binary_cross_entropy_with_logits(
        real_preds, torch.ones_like(real_preds)
    )
    adversarial_loss_fake = F.binary_cross_entropy_with_logits(
        fake_preds, torch.zeros_like(fake_preds)
    )
    adversarial_loss = (adversarial_loss_real + adversarial_loss_fake) / 2

    # Perceptual L1 Loss
    l1_loss = F.l1_loss(generated_images, real_images)

    # Rescale from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2
    real_images = (real_images + 1) / 2

    assert torch.all(
        (generated_images >= 0) & (generated_images <= 1)
    ), "Generated image has pixel values out of the [0, 1] range."
    assert torch.all(
        (real_images >= 0) & (real_images <= 1)
    ), "Real image has pixel values out of the [0, 1] range."

    # SSIM Loss (1 - SSIM)
    ssim_loss = 1 - piq.ssim(generated_images, real_images, data_range=1.0)

    # PSNR Loss
    psnr_value = piq.psnr(generated_images, real_images, data_range=1.0)

    psnr_loss = -torch.log(psnr_value + 1e-7)

    # Weighted sum of all losses
    combined_loss = (
        (alpha * adversarial_loss)
        + (beta * l1_loss)
        + (gamma * ssim_loss)
        + (delta * psnr_loss)
    )

    return combined_loss


def total_variation_loss(image):
    # Assuming image of shape [1, height, width]
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    print(f"Image shape after adding batch dimension (if needed): {image.shape}")

    # Calculate TV loss
    tv_loss_x = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    tv_loss_y = torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))

    tv_loss = tv_loss_x + tv_loss_y
    print(f"Total Variation Loss: {tv_loss.item()}")

    return tv_loss


def GDNLoss(generated_HR, true_LR, estimated_blur_kernel, lambda_tv):
    """
    Combined loss that includes blur kernel estimation and total variation loss.
    """
    # Blur Kernel Estimation loss
    loss_blur = F.mse_loss(true_LR, estimated_blur_kernel)
    print(f"Blur Kernel Estimation Loss: {loss_blur.item()}")

    # Total Variation Loss for edge preservation in HR reconstruction
    tv_loss = total_variation_loss(generated_HR)
    print(f"Total Variation Loss: {tv_loss}")

    # Combined loss
    total_loss = loss_blur + lambda_tv * tv_loss
    print(f"Combined GDN Loss: {total_loss.item()}")

    return total_loss
