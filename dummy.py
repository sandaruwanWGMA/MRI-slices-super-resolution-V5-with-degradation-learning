import torch
from model.RRDB_architecture import RRDBNet


# Load the model architecture as currently defined
model = RRDBNet(
    in_nc=1, out_nc=1, nf=64, nb=23, gc=32
)  # Adjusted for grayscale input/output

# Load the pre-trained weights
pretrained_weights = torch.load("RRDB_ESRGAN_x4.pth", map_location=torch.device("cpu"))

# Manually adjust the weights for the first and last convolutional layers
pretrained_weights["conv_first.weight"] = pretrained_weights["conv_first.weight"][
    :, 0:1, :, :
].clone()
pretrained_weights["conv_last.weight"] = pretrained_weights["conv_last.weight"][
    0:1, :, :, :
].clone()
pretrained_weights["conv_last.bias"] = pretrained_weights["conv_last.bias"][0:1].clone()

# Try to load the adjusted weights into your model
try:
    model.load_state_dict(pretrained_weights)
    print("Weights loaded successfully.")
except RuntimeError as e:
    print("Failed to load weights:", e)

# Proceed with the rest of your model setup or evaluation
