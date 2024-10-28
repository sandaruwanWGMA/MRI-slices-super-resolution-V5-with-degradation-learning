import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from SRUNet import SRUNet

image_size = 256
model = SRUNet(
    image_size=image_size, in_channels=1, out_channels=1, freeze_encoder=True
)

# Input image
# x = torch.randn(1, 1, image_size, image_size)  # Example input with batch size 1, grayscale

# print(x)

# Load the model weights from the .pth file
model.load_state_dict(
    torch.load("model/final_models-SRUNet_final.pth", map_location=torch.device("cpu"))
)

model.eval()

# Move the input tensor to the same device as the model
array = np.array([[input_data[0]]], dtype=np.float32)
y = torch.from_numpy(array)
input_tensor = y

# Pass the input through the model in evaluation mode
with torch.no_grad():
    output = model(input_tensor)

print(output.shape)  # Should print torch.Size([1, 1, 256, 256])


for i in range(1, 150):
    # Move the input tensor to the same device as the model
    array = np.array([[input_data[i]]], dtype=np.float32)
    y = torch.from_numpy(array)
    input_tensor = y

    # Pass the input through the model in evaluation mode
    with torch.no_grad():
        output1 = model(input_tensor)

    output = torch.cat((output, output1), dim=1)

print(output.shape)


# # Convert the output2 tensor to a NumPy array for visualization
# output2_image = output2.squeeze().cpu().numpy()

# # Plot the output2 image
# plt.imshow(output2_image, cmap='gray')
# plt.colorbar()  # Optional: adds a colorbar to show pixel values
# plt.title('SRUNet Output2')
# plt.show()

# Assuming `output` is the tensor with shape [1, 150, 256, 256]
output_np = output.squeeze().cpu().numpy()
print(output_np.shape)  # Should print (150, 256, 256)


input_header = input.header

output_img = nib.Nifti1Image(output_np, input.affine, input_header)
nib.save(output_img, "output_image2.nii")
