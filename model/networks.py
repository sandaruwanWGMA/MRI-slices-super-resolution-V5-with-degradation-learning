import torch
import torch.nn as nn
import torchvision.models as models
import torch.fft
import torch.nn.functional as F
from torchvision.models import VGG19_Weights


class CustomMiniPatchGAN(nn.Module):
    def __init__(self):
        super(CustomMiniPatchGAN, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        # Initial convolution layer to reduce the channels from 512 to 256
        self.initial = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        # Intermediate layers including downsampling and normalization
        self.middle = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                256, 128, kernel_size=3, stride=2, padding=1
            ),  # Downsampling to 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1
            ),  # Maintaining size 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final convolution layer to get to 1 output channel
        self.final = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.initial(x)  # Output: [1, 256, 8, 8]
        x = self.middle(x)
        x = self.final(x)
        return x


class SingleChannelVGG(nn.Module):
    def __init__(self):
        super(SingleChannelVGG, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        # Load original VGG features
        original_vgg_features = vgg19.features

        # Modify the first convolutional layer
        first_conv_layer = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # Copy the weights from the first three channels of the pretrained model
        first_conv_layer.weight.data = (
            original_vgg_features[0].weight.data[:, 0:1, :, :].clone()
        )
        first_conv_layer.bias.data = original_vgg_features[0].bias.data.clone()

        # Replace the first conv layer in the features list
        modified_features = nn.Sequential(
            first_conv_layer, *list(original_vgg_features.children())[1:]
        )

        # Assign modified features to the class variable
        self.vgg_layers = modified_features

        for i in range(10):  # Freezing the first 5 layers, for example
            for param in self.vgg_layers[i].parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.vgg_layers(x)
        return x


class DoubleChannelVGG(nn.Module):
    def __init__(self):
        super(DoubleChannelVGG, self).__init__()
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        # Load original VGG features
        original_vgg_features = vgg19.features

        # Modify the first convolutional layer for 2 input channels
        first_conv_layer = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)

        # Initialize weights for the two input channels by copying weights from the original single-channel weights
        # We'll duplicate the weights across two channels or initialize the second channel with random weights
        first_conv_layer.weight.data[:, 0:1, :, :] = (
            original_vgg_features[0].weight.data[:, 0:1, :, :].clone()
        )
        first_conv_layer.weight.data[:, 1:2, :, :] = (
            original_vgg_features[0].weight.data[:, 0:1, :, :].clone()
        )
        first_conv_layer.bias.data = original_vgg_features[0].bias.data.clone()

        # Replace the first conv layer in the features list
        modified_features = nn.Sequential(
            first_conv_layer, *list(original_vgg_features.children())[1:]
        )

        # Assign modified features to the class variable
        self.vgg_layers = modified_features

    def forward(self, x):
        # Pass input through modified VGG layers
        return self.vgg_layers(x)


class GeneratorFrequencyFilter(nn.Module):
    def __init__(self, image_size):
        super(GeneratorFrequencyFilter, self).__init__()
        # Create a Gaussian-sinc combined filter in frequency domain
        self.register_buffer("filter", self.create_gaussian_sinc_filter(image_size))

    def forward(self, x):
        # Apply FFT
        x_fft = torch.fft.fft2(x)
        # Filter in the frequency domain
        filtered_fft = x_fft * self.filter
        # Apply Inverse FFT
        x_filtered = torch.fft.ifft2(
            filtered_fft
        ).real  # Use the real part for image data
        return x_filtered

    def create_gaussian_sinc_filter(self, size):
        # filter: Modify as per actual sinc and Gaussian parameters
        sigma = size / 6.0  # Standard deviation for Gaussian
        x = torch.arange(size).float() - size / 2
        x, y = torch.meshgrid(x, x)
        r = torch.sqrt(x**2 + y**2)
        sinc = torch.sin(r) / r
        sinc[r == 0] = 1  # Handling NaN at the center
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return (sinc * gaussian).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)


# GDNGaussianSincFilter class with filter resizing
class GDNGaussianSincFilter(nn.Module):
    def __init__(self, size, sigma, sinc_scale):
        super(GDNGaussianSincFilter, self).__init__()
        if size % 2 == 0:
            size += 1
        self.register_buffer("filter", self.create_filter(size, sigma, sinc_scale))

    def forward(self, x):
        # Apply FFT
        x_fft = torch.fft.fft2(x)

        # Resize filter to match input dimensions
        resized_filter = F.interpolate(
            self.filter,
            size=(x_fft.shape[2], x_fft.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Apply the filter in the frequency domain
        filtered_fft = x_fft * resized_filter

        # Apply Inverse FFT
        x_filtered = torch.fft.ifft2(filtered_fft).real
        return x_filtered

    def create_filter(self, size, sigma, sinc_scale):
        ind = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float32)
        X, Y = torch.meshgrid(ind, ind, indexing="ij")
        radius = torch.sqrt(X**2 + Y**2)
        sinc = torch.sin(sinc_scale * radius) / (sinc_scale * radius)
        sinc[radius == 0] = 1
        gaussian = torch.exp(-0.5 * (X**2 + Y**2) / sigma**2)
        combined_filter = sinc * gaussian
        combined_filter /= combined_filter.sum()
        return combined_filter.view(1, 1, size, size)
