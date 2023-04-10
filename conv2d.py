import torch
import torch.nn as nn

# manual convolution

def conv2d(input_tensor, kernel, padding=0, stride=1):
    assert input_tensor.dim() == 2, "Input tensor should be 2D"
    assert kernel.dim() == 2, "Kernel should be 2D"
    assert padding >= 0, "Padding should be non-negative"
    assert stride > 0, "Stride should be positive"

    input_height, input_width = input_tensor.size()
    kernel_height, kernel_width = kernel.size()

    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    output_tensor = torch.zeros(output_height, output_width)
    padded_input = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding))

    for i in range(0, output_height):
        for j in range(0, output_width):
            x_start = i * stride
            x_end = x_start + kernel_height
            y_start = j * stride
            y_end = y_start + kernel_width

            window = padded_input[x_start:x_end, y_start:y_end]
            output_tensor[i, j] = torch.sum(window * kernel)

    return output_tensor

# Example usage
input_tensor = torch.tensor([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=torch.float32)

kernel = torch.tensor([
    [1, 0],
    [0, -1]
], dtype=torch.float32)

output = conv2d(input_tensor, kernel, padding=1, stride=1)
print(output)
