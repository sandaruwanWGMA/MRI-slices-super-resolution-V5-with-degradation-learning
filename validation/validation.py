import torch
from torch.utils.data import DataLoader
from ..model.SRUNet import SRUNet
from data.dataloader import MRIDataset
from options.train_options import TrainOptions
from skimage import feature, metrics
import lpips
import numpy as np

# Initialize the model and DataLoader
opt = TrainOptions().parse()
val_data = "dataset/val_filenames.txt"
val_dataset = MRIDataset(val_data)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False
)  # Batch size is 1 for simplicity

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model setup
model = SRUNet(image_size=256, in_channels=1, out_channels=1, freeze_encoder=True)
model.load_state_dict(
    torch.load("model/final_models-SRUNet_final.pth", map_location=device)
)
model.to(device)
model.eval()

# Initialize Perceptual Loss
perceptual_loss = lpips.LPIPS(net="vgg").to(device)

# Initialize metrics
total_psnr = 0
total_ssim = 0
total_edge_accuracy = 0
total_perceptual_loss = 0
num_samples = 0


def edge_accuracy(pred, target):
    # Edge detection must be run on CPU using skimage
    edges_pred = feature.canny(pred)
    edges_true = feature.canny(target)
    return np.mean(edges_pred == edges_true)


# Iterate over the dataset
for i, data in enumerate(val_loader, 0):
    low_res_image, high_res_image = data[0].to(device), data[1].to(device)

    with torch.no_grad():
        output = model(low_res_image)

    # Convert tensors to numpy for skimage (run on CPU)
    output_np = output.squeeze().cpu().numpy()
    high_res_np = high_res_image.squeeze().cpu().numpy()

    # Calculate PSNR and SSIM
    psnr = metrics.peak_signal_noise_ratio(
        high_res_np, output_np, data_range=high_res_np.max() - high_res_np.min()
    )
    ssim = metrics.structural_similarity(
        high_res_np, output_np, data_range=high_res_np.max() - high_res_np.min()
    )
    edge_acc = edge_accuracy(output_np, high_res_np)

    # Calculate Perceptual Loss
    p_loss = perceptual_loss(output, high_res_image).mean().item()

    total_psnr += psnr
    total_ssim += ssim
    total_edge_accuracy += edge_acc
    total_perceptual_loss += p_loss
    num_samples += 1

# Calculate averages
average_psnr = total_psnr / num_samples
average_ssim = total_ssim / num_samples
average_edge_accuracy = total_edge_accuracy / num_samples
average_perceptual_loss = total_perceptual_loss / num_samples

print(f"Average PSNR: {average_psnr}")
print(f"Average SSIM: {average_ssim}")
print(f"Average Edge Accuracy: {average_edge_accuracy}")
print(f"Average Perceptual Loss: {average_perceptual_loss}")
