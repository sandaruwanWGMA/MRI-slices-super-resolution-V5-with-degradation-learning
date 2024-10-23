import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)


class CustomDeepLab(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        freeze_backbone=False,
        freeze_classifier=False,
    ):
        super(CustomDeepLab, self).__init__()
        # Load the pre-trained DeepLabV3 model with the appropriate weights
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.deeplab = deeplabv3_resnet101(weights=weights)

        # Modify the first convolution layer to accept custom number of input channels
        self.deeplab.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,  # Use in_channels parameter
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Replace all BatchNorm2d layers with GroupNorm
        self._replace_batchnorm(self.deeplab)

        # Replace in-place ReLU with out-of-place ReLU
        self._replace_relu(self.deeplab)

        # Modify the classifier to output num_classes channels
        self.deeplab.classifier[-1] = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,  # Use num_classes parameter
            kernel_size=1,
        )

        # If aux classifier is used, adjust it as well
        if self.deeplab.aux_classifier is not None:
            self.deeplab.aux_classifier[-1] = nn.Conv2d(
                in_channels=256,
                out_channels=num_classes,  # Use num_classes parameter
                kernel_size=1,
            )

        # Freeze parameters based on options
        if freeze_backbone:
            for param in self.deeplab.backbone.parameters():
                param.requires_grad = False

        if freeze_classifier:
            for param in self.deeplab.classifier.parameters():
                param.requires_grad = False

    def _replace_batchnorm(self, module):
        """
        Recursively replace all nn.BatchNorm2d layers with nn.GroupNorm.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_features = child.num_features
                # Choose a reasonable number of groups
                num_groups = 32 if num_features >= 32 else max(1, num_features // 2)
                # Replace with GroupNorm
                new_layer = nn.GroupNorm(
                    num_groups=num_groups, num_channels=num_features
                )
                setattr(module, name, new_layer)
            else:
                # Recursively apply to child modules
                self._replace_batchnorm(child)

    def _replace_relu(self, module):
        """
        Recursively replace all in-place ReLU activations with out-of-place ReLU.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                if child.inplace:
                    # Replace in-place ReLU with out-of-place ReLU
                    setattr(module, name, nn.ReLU(inplace=False))
            else:
                # Recursively apply to child modules
                self._replace_relu(child)

    def forward(self, x):
        return self.deeplab(x)


# # Create an instance of the custom model
# model = CustomDeepLab()


# # Create a dummy input tensor with the shape [1, 1, 256, 256]
# dummy_input = torch.randn(1, 1, 256, 256)

# # Pass the dummy input through the model
# output = model(dummy_input)

# # Access the tensor output from the output dictionary
# output_tensor = output["out"]

# # Print the shape of the output tensor
# print(output_tensor.shape)
