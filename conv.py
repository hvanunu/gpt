import torch
import torchvision.transforms as transforms
from PIL import Image

# Load an image from file
image_path = 'superman.webp'
image = Image.open(image_path)

# Define the transformation pipeline for grayscale
transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image to a size appropriate for your model (e.g., 32x32)
    transforms.Grayscale(),  # Convert the image to grayscale (single channel)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Apply the transformations to the image
processed_image = transform_pipeline(image)

# Add a batch dimension (required for input to the convolutional neural network)
input_image = processed_image.unsqueeze(0)  # Resulting shape: (1, 1, 32, 32)

# Define the 3x3 convolution kernel
kernel = torch.tensor([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=torch.float32)

# Add the channel and batch dimensions to the kernel
kernel = kernel.view(1, 1, 3, 3)

# Perform the convolution
convoluted_image = torch.nn.functional.conv2d(input_image, kernel, padding=1)

# Remove the batch dimension (for display purposes)
convoluted_image = convoluted_image.squeeze(0)

print(convoluted_image.shape)

# Display the image
import matplotlib.pyplot as plt
plt.imshow(convoluted_image[0], cmap='gray')
plt.show()
