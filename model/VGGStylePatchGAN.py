import torch
import torch.nn as nn
import torchvision.models as models
from .networks import SingleChannelVGG, DoubleChannelVGG, CustomMiniPatchGAN


class VGGStylePatchGAN(nn.Module):
    def __init__(self, patch_size):
        super(VGGStylePatchGAN, self).__init__()

        # Single channel pre-trained VGG
        self.vgg_layers = SingleChannelVGG()

        # Additional convolutional layers to transform VGG features for patch classification
        self.mini_patch_gan = CustomMiniPatchGAN()

        # Sigmoid for patch-wise real/fake probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extract perceptual features using VGG layers
        features = self.vgg_layers(x)

        # Process through additional conv layers for patch-based classification
        patch_predictions = self.mini_patch_gan(features)

        # Apply sigmoid to get probability outputs per patch
        return self.sigmoid(patch_predictions)


# Example usage:
discriminator = VGGStylePatchGAN(patch_size=70)
# input_image = torch.randn(1, 1, 256, 256)
# output = discriminator(input_image)
# print(output)  # Should reflect patch-based output


def count_params(model):
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_count = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    return trainable_count, non_trainable_count


trainable_params, non_trainable_params = count_params(discriminator)
print("Trainable params:", trainable_params)
print("Non-trainable params:", non_trainable_params)
