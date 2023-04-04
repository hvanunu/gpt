import torch
import torchvision.transforms as transforms
from PIL import Image

# Load an image from file
image_path = 'superman.webp'
image = Image.open(image_path)

# Define the transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image to a size appropriate for your model (e.g., 32x32)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Apply the transformations to the image
processed_image = transform_pipeline(image)
print("the shape of an image" , processed_image.shape)

# Add a batch dimension (required for input to the convolutional neural network)
processed_image = processed_image.unsqueeze(0)  #
print("shape of a timeline", processed_image.shape)

# create a tensor of 3X3 filter with all ones
filter = torch.ones(3, 3, 3)
print("filter shape", filter.shape)

# multiply the filter with the image
output = torch.nn.functional.conv2d(processed_image, filter, padding=1)
print("output shape", output.shape)
