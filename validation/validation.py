import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.SRUNet import SRUNet


image_size = 256
model = SRUNet(
    image_size=image_size, in_channels=1, out_channels=1, freeze_encoder=True
)

# Input image
x = torch.randn(
    1, 1, image_size, image_size
)  # Example input with batch size 1, grayscale

# Load the model weights from the .pth file
model.load_state_dict(torch.load("validation/final_models-SRUNet_final.pth"))

model.eval()

# Move the input tensor to the same device as the model
input_tensor = x

# Pass the input through the model in evaluation mode
with torch.no_grad():
    output = model(input_tensor)

# Print or process the output as needed
print(output)
