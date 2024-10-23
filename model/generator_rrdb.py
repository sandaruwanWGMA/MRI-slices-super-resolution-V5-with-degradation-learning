import torch
import torch.nn as nn
from networks import GeneratorFrequencyFilter
from RRDB_architecture import RRDBNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(
        self, in_nc, out_nc, nf, nb, gc=32, image_size=256, pretrained_path=None
    ):
        super(Generator, self).__init__()
        self.freq_filter = GeneratorFrequencyFilter(image_size)

        # Initialize RRDBNet
        self.rrdb_net = RRDBNet(in_nc, out_nc, nf, nb, gc)

        # Load pre-trained weights if provided
        if pretrained_path:
            self.load_rrdb_weights(pretrained_path)

        # Downsampling layers to reduce from 1024x1024 to 256x256
        # self.downsample1 = nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=2, padding=1)
        # self.downsample2 = nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=2, padding=1)

        # Adaptive pooling to ensure exact 256x256 size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((256, 256))

        self.relu = nn.ReLU(inplace=True)

    def load_rrdb_weights(self, path):
        # Load pre-trained weights
        print(f"Loading pre-trained RRDBNet weights from {path}")

        pretrained_weights = torch.load(path)

        # Manually adjust the weights for the first and last convolutional layers
        pretrained_weights["conv_first.weight"] = pretrained_weights[
            "conv_first.weight"
        ][:, 0:1, :, :].clone()
        pretrained_weights["conv_last.weight"] = pretrained_weights["conv_last.weight"][
            0:1, :, :, :
        ].clone()
        pretrained_weights["conv_last.bias"] = pretrained_weights["conv_last.bias"][
            0:1
        ].clone()

    def forward(self, x):
        x = self.freq_filter(x)
        x = self.rrdb_net(x)
        # x = self.relu(self.downsample1(x))
        # x = self.relu(self.downsample2(x))
        # x = self.adaptive_pool(x)  # Final downsampling to exactly 256x256
        return x


image_size = 256  # Define the image size used for the filter
in_nc = 1  # Number of input channels (e.g., RGB)
out_nc = 1  # Number of output channels (should match input if same space)
nf = 64  # Number of features
nb = 23  # Number of blocks

# Create generator
generator = Generator(
    in_nc, out_nc, nf, nb, image_size=image_size, pretrained_path="RRDB_ESRGAN_x4.pth"
)

# Dummy input - replace with actual data
input_image = torch.randn(1, in_nc, image_size, image_size)  # Random noise image

# Get the output
output_image = generator(input_image)
print(output_image.shape)
