import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .networks import GeneratorFrequencyFilter


class SRUNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels=1,
        out_channels=1,
        freeze_encoder=True,
        unfreeze_layers=None,
    ):
        super(SRUNet, self).__init__()

        # Initialize frequency filter
        self.frequency_filter = GeneratorFrequencyFilter(image_size)

        # Initialize UNet with EfficientNet-b3 encoder
        self.unet = smp.Unet(
            encoder_name="efficientnet-b3",  # Suitable for MRI super-resolution
            encoder_weights="imagenet",  # Pre-trained weights
            in_channels=in_channels,  # Typically 1 for grayscale MRI images
            classes=out_channels,  # Number of output channels
        )

        # freeze all encoder layers
        if freeze_encoder:
            for name, param in self.unet.encoder.named_parameters():
                param.requires_grad = False

            # Unfreeze specified layers
            if unfreeze_layers is not None:
                for layer_name in unfreeze_layers:
                    if hasattr(self.unet.encoder, layer_name):
                        layer = getattr(self.unet.encoder, layer_name)
                        for param in layer.parameters():
                            param.requires_grad = True
                    else:
                        print(f"Layer {layer_name} not found in encoder.")

        self.tanh = nn.Tanh()

    def forward(self, x):
        x_filtered = self.frequency_filter(x)
        x_unet = self.unet(x_filtered)
        return self.tanh(x_unet)


# Define image size and instantiate model
# image_size = 256
# model = SRUNet(
#     image_size=image_size, in_channels=1, out_channels=1, freeze_encoder=True
# )

# # Input image
# x = torch.randn(
#     1, 1, image_size, image_size
# )  # Example input with batch size 1, grayscale

# # Forward pass
# output = model(x)
# print(output)
